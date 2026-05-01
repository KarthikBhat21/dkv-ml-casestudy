import argparse
import json
import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET = "default"


# ── Argument parsing ───────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Train credit default classifier.")
    parser.add_argument("--input_train",  type=str, default=None)
    parser.add_argument("--output_model", type=str, default=None)
    args = parser.parse_args()

    if args.input_train is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.input_train  = os.path.join(BASE_DIR, "artifacts", "preprocessed_data", "train.csv")
        args.output_model = os.path.join(BASE_DIR, "artifacts", "models")

    return args


# ── Data loading ───────────────────────────────────────────────────────────────

def load_training_data(path: str):
    logger.info("Loading training data from: %s", path)
    df = pd.read_csv(path)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]
    logger.info("Training set: %d rows, %d features", *X.shape)
    logger.info("Default rate: %.2f%%", y.mean() * 100)
    return X, y


# ── Params loading ─────────────────────────────────────────────────────────────

def load_params() -> dict:
    """
    Load hyperparameter search space from params.yaml.
    Looks in repo root (two levels up from src/).
    Falls back to empty dict if file not found.
    """
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params_path = os.path.join(base_dir, "params.yaml")

    if not os.path.exists(params_path):
        logger.warning("params.yaml not found at %s — using empty grids.", params_path)
        return {}

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    logger.info("Loaded params.yaml from: %s", params_path)
    return params


# ── Model definitions ──────────────────────────────────────────────────────────

def get_candidate_models(scale_pos_weight: float) -> dict:
    """
    7 candidate classifiers with default parameters.
    scale_pos_weight passed to XGBoost for class imbalance.
    All others use class_weight='balanced' where supported.
    """
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


# ── Phase 1: CV comparison with default params ─────────────────────────────────

def compare_models(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> dict:
    """
    Evaluate all candidate models using cross-validation (ROC-AUC).
    Returns dict of {model_name: {"mean_auc": float, "std_auc": float}}
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 — Comparing %d models with default parameters", len(models))
    logger.info("=" * 60)

    results = {}

    for name, model in models.items():
        logger.info("Evaluating: %s ...", name)
        try:
            scores = cross_val_score(
                model, X, y,
                cv=cv_folds,
                scoring="roc_auc",
                n_jobs=-1,
            )
            mean_auc = scores.mean()
            std_auc  = scores.std()
            results[name] = {
                "mean_auc": round(mean_auc, 4),
                "std_auc":  round(std_auc, 4),
            }
            logger.info(
                "  %-22s CV ROC-AUC: %.4f ± %.4f",
                name, mean_auc, std_auc,
            )
        except Exception as e:
            logger.warning("  %s failed during CV: %s — skipping.", name, str(e))

    # Sort and display leaderboard
    logger.info("")
    logger.info("── Model Leaderboard (default params) ──")
    for rank, (name, res) in enumerate(
        sorted(results.items(), key=lambda x: x[1]["mean_auc"], reverse=True), 1
    ):
        logger.info(
            "  #%d %-22s AUC: %.4f ± %.4f",
            rank, name, res["mean_auc"], res["std_auc"],
        )

    return results


# ── Phase 2: Hyperparameter tuning on winner only ──────────────────────────────

def tune_best_model(
    best_name: str,
    best_model,
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
) -> tuple:
    """
    Run RandomizedSearchCV on the winning model using the
    search space defined in params.yaml.
    Returns (tuned_model, best_params, best_cv_auc).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2 — Hyperparameter tuning: %s", best_name)
    logger.info("=" * 60)

    # Load search config
    search_cfg  = params.get("random_search", {})
    n_iter      = search_cfg.get("n_iter",       20)
    cv_folds    = search_cfg.get("cv_folds",      5)
    scoring     = search_cfg.get("scoring",       "roc_auc")
    random_state= search_cfg.get("random_state",  42)

    # Load param grid for winner
    param_grid = params.get(best_name, {})

    if not param_grid:
        logger.warning(
            "No param grid found for '%s' in params.yaml. "
            "Skipping tuning — using default parameters.",
            best_name,
        )
        return best_model, {}, None

    # Convert null → None for sklearn compatibility
    for key, values in param_grid.items():
        param_grid[key] = [None if v is None else v for v in values]

    logger.info("Search space: %s", param_grid)
    logger.info(
        "Running RandomizedSearchCV — n_iter=%d, cv=%d, scoring=%s",
        n_iter, cv_folds, scoring,
    )

    search = RandomizedSearchCV(
        estimator=best_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_folds,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)

    best_params   = search.best_params_
    best_cv_auc   = round(search.best_score_, 4)
    tuned_model   = search.best_estimator_

    logger.info("Best params found : %s", best_params)
    logger.info("Best CV ROC-AUC   : %.4f", best_cv_auc)

    return tuned_model, best_params, best_cv_auc


# ── MLflow setup ───────────────────────────────────────────────────────────────

def setup_mlflow():
    """
    On Azure ML, tracking URI is auto-injected via AZUREML_RUN_ID.
    Locally, use SQLite to avoid Windows path-with-spaces bug.
    """
    if "AZUREML_RUN_ID" not in os.environ:
        base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mlflow_db = os.path.join(base_dir, "artifacts", "mlruns", "mlflow.db")
        os.makedirs(os.path.dirname(mlflow_db), exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        mlflow.set_experiment("dkv-credit-default-local")
        logger.info("MLflow tracking: SQLite (local mode)")
    else:
        logger.info("MLflow tracking: Azure ML (cloud mode)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.output_model, exist_ok=True)

    # Load data
    X_train, y_train = load_training_data(args.input_train)

    # Class imbalance ratio for XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    logger.info(
        "Class ratio — neg: %d | pos: %d | scale_pos_weight: %.2f",
        neg, pos, scale_pos_weight,
    )

    # Load hyperparameter search space
    params = load_params()

    # MLflow setup
    setup_mlflow()

    with mlflow.start_run(run_name="multi_model_training"):

        # ── Phase 1: Compare all 7 models with default params ─────────────────
        candidates = get_candidate_models(scale_pos_weight)
        cv_results = compare_models(
            models=candidates,
            X=X_train,
            y=y_train,
            cv_folds=params.get("random_search", {}).get("cv_folds", 5),
        )

        # Log all CV scores to MLflow
        for name, res in cv_results.items():
            mlflow.log_metrics({
                f"{name}_default_cv_auc_mean": res["mean_auc"],
                f"{name}_default_cv_auc_std":  res["std_auc"],
            })

        # Pick the winner
        best_name = max(cv_results, key=lambda n: cv_results[n]["mean_auc"])
        best_default_auc = cv_results[best_name]["mean_auc"]

        logger.info("")
        logger.info("Winner (default params): %s — AUC: %.4f", best_name, best_default_auc)
        mlflow.log_param("phase1_winner",          best_name)
        mlflow.log_metric("phase1_best_default_auc", best_default_auc)

        # ── Phase 2: Tune winner with RandomizedSearchCV ───────────────────────
        tuned_model, best_params, tuned_auc = tune_best_model(
            best_name=best_name,
            best_model=candidates[best_name],
            X=X_train,
            y=y_train,
            params=params,
        )

        # Log tuning results
        if best_params:
            mlflow.log_params({f"tuned_{k}": v for k, v in best_params.items()})
        if tuned_auc:
            mlflow.log_metric("phase2_tuned_cv_auc", tuned_auc)

        auc_improvement = round((tuned_auc or best_default_auc) - best_default_auc, 4)
        logger.info("AUC improvement from tuning: +%.4f", auc_improvement)
        mlflow.log_metric("auc_improvement_from_tuning", auc_improvement)

        # ── Phase 3: Retrain on full training set ──────────────────────────────
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "PHASE 3 — Retraining %s with best params on full training set",
            best_name,
        )
        logger.info("=" * 60)

        final_model = tuned_model
        final_model.fit(X_train, y_train)

        train_auc = roc_auc_score(
            y_train, final_model.predict_proba(X_train)[:, 1]
        )
        train_acc = final_model.score(X_train, y_train)

        mlflow.log_metric("final_train_auc",      round(train_auc, 4))
        mlflow.log_metric("final_train_accuracy",  round(train_acc, 4))

        logger.info("Final train AUC:      %.4f", train_auc)
        logger.info("Final train accuracy: %.4f", train_acc)

        # ── Phase 4: Save model + metadata ────────────────────────────────────
        model_path = os.path.join(args.output_model, "model.pkl")
        joblib.dump(final_model, model_path)
        logger.info("Model saved → %s", model_path)

        metadata = {
            "best_model":            best_name,
            "selection_metric":      "roc_auc",
            "phase1_default_cv_auc": best_default_auc,
            "phase2_tuned_cv_auc":   tuned_auc or best_default_auc,
            "auc_improvement":       auc_improvement,
            "best_params":           best_params,
            "final_train_auc":       round(train_auc, 4),
            "final_train_accuracy":  round(train_acc, 4),
            "scale_pos_weight":      round(scale_pos_weight, 4),
            "all_models_default_cv": cv_results,
        }
        metadata_path = os.path.join(args.output_model, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata saved → %s", metadata_path)

        # Log both files as MLflow artifacts
        mlflow.log_artifact(model_path,    artifact_path="model")
        mlflow.log_artifact(metadata_path, artifact_path="model")

        # ── Summary ────────────────────────────────────────────────────────────
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE ✓")
        logger.info("  Winner model       : %s", best_name)
        logger.info("  Default CV AUC     : %.4f", best_default_auc)
        logger.info("  Tuned CV AUC       : %.4f", tuned_auc or best_default_auc)
        logger.info("  AUC improvement    : +%.4f", auc_improvement)
        logger.info("  Final train AUC    : %.4f", train_auc)
        logger.info("  Model saved at     : %s", model_path)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()