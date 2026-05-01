import logging
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET = "default"


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate credit default classifier.")
    parser.add_argument("--input_test",  type=str, default=None)
    parser.add_argument("--input_model", type=str, default=None)
    parser.add_argument("--output_dir",  type=str, default=None)
    args = parser.parse_args()

    # Fall back to local paths when not running inside Azure ML
    if args.input_test is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.input_test  = os.path.join(BASE_DIR, "artifacts", "preprocessed_data", "test.csv")
        args.input_model = os.path.join(BASE_DIR, "artifacts", "models")
        args.output_dir  = os.path.join(BASE_DIR, "artifacts", "evaluation")

    return args


def plot_confusion_matrix(cm: np.ndarray, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    classes    = ["No Default (0)", "Default (1)"]
    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11, fontweight="bold")

    ax.set_ylabel("True label", fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_title("Confusion Matrix — Credit Default Classifier", fontsize=11)
    fig.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", path)
    return path


def setup_mlflow():
    """
    On Azure ML, tracking URI is auto-injected.
    Locally, use SQLite to avoid Windows path-with-spaces bug.
    """
    if "AZUREML_RUN_ID" not in os.environ:
        BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MLFLOW_DB = os.path.join(BASE_DIR, "artifacts", "mlruns", "mlflow.db")
        os.makedirs(os.path.dirname(MLFLOW_DB), exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
        mlflow.set_experiment("dkv-credit-default-local")
        logger.info("MLflow tracking: SQLite (local mode)")
    else:
        logger.info("MLflow tracking: Azure ML (cloud mode)")


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading test data from: %s", args.input_test)
    df     = pd.read_csv(args.input_test)
    X_test = df.drop(columns=[TARGET])
    y_test = df[TARGET]
    logger.info("Test set: %d rows", len(y_test))

    logger.info("Loading model from: %s", args.input_model)
    model = joblib.load(os.path.join(args.input_model, "model.pkl"))

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy  = accuracy_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_pred_prob)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)

    logger.info("=== Evaluation Results ===")
    logger.info("Accuracy  : %.4f", accuracy)
    logger.info("ROC-AUC   : %.4f", auc)
    logger.info("F1 Score  : %.4f", f1)
    logger.info("Precision : %.4f", precision)
    logger.info("Recall    : %.4f", recall)

    report = classification_report(
        y_test, y_pred, target_names=["No Default", "Default"]
    )
    logger.info("\n%s", report)

    # ── MLflow ────────────────────────────────────────────────────────────────
    setup_mlflow()

    with mlflow.start_run():
        mlflow.log_metrics({
            "test_accuracy":  round(accuracy, 4),
            "test_roc_auc":   round(auc, 4),
            "test_f1":        round(f1, 4),
            "test_precision": round(precision, 4),
            "test_recall":    round(recall, 4),
        })

        cm      = confusion_matrix(y_test, y_pred)
        cm_path = plot_confusion_matrix(cm, args.output_dir)
        mlflow.log_artifact(cm_path, artifact_path="evaluation")

        report_path = os.path.join(args.output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="evaluation")

    logger.info("Evaluation complete ✓  Artefacts → %s", args.output_dir)


if __name__ == "__main__":
    main()