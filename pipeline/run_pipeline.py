import json
import logging
import os
import tempfile
import time

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Environment, Model
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Workspace configuration ────────────────────────────────────────────────────
SUBSCRIPTION_ID  = "b17046df-01cb-41d8-966c-a496e8fbef0e"
RESOURCE_GROUP   = "rg-dkv-ml-casestudy"
WORKSPACE_NAME   = "dkv-ml-workspace"
COMPUTE_CLUSTER  = "dkv-cpu-cluster"
ENVIRONMENT_NAME = "dkv-ml-env"
EXPERIMENT_NAME  = "dkv-credit-default-experiment"
MODEL_NAME       = "credit-default-best-model"

# Paths — code=REPO_ROOT so params.yaml is included in upload
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDA_FILE = os.path.join(REPO_ROOT, "environments", "conda.yaml")


# ── Client ────────────────────────────────────────────────────────────────────

def get_ml_client() -> MLClient:
    logger.info("Connecting to Azure ML workspace: %s", WORKSPACE_NAME)
    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    logger.info("Connected successfully ✓")
    return client


# ── Environment ───────────────────────────────────────────────────────────────

def get_or_create_environment(ml_client: MLClient) -> Environment:
    try:
        env = ml_client.environments.get(ENVIRONMENT_NAME, label="latest")
        logger.info("Reusing existing environment: %s", ENVIRONMENT_NAME)
        return env
    except Exception:
        logger.info("Registering new environment: %s", ENVIRONMENT_NAME)
        env = Environment(
            name=ENVIRONMENT_NAME,
            description="DKV ML case study — multi-model + RandomizedSearchCV + azureml-mlflow",
            conda_file=CONDA_FILE,
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        )
        env = ml_client.environments.create_or_update(env)
        logger.info(
            "Environment registered: %s (version %s)", env.name, env.version
        )
        return env


# ── Pipeline definition ───────────────────────────────────────────────────────

def build_pipeline(env: Environment):

    # ── Step 1: Preprocess ─────────────────────────────────────────────────────
    preprocess_component = command(
        name="preprocess",
        display_name="Step 1 — Preprocess",
        description=(
            "Load raw XLS, clean, engineer features, "
            "scale with StandardScaler, split 80/20 train/test."
        ),
        command=(
            "python src/preprocess.py"
            " --input_data ${{inputs.raw_data}}"
            " --output_train ${{outputs.train_data}}/train.csv"
            " --output_test  ${{outputs.test_data}}/test.csv"
        ),
        inputs={
            "raw_data": Input(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT,
            ),
        },
        outputs={
            "train_data": Output(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RW_MOUNT,
            ),
            "test_data": Output(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RW_MOUNT,
            ),
        },
        environment=env,
        compute=COMPUTE_CLUSTER,
        code=REPO_ROOT,        # ← whole repo uploaded so params.yaml is included
    )

    # ── Step 2: Train ──────────────────────────────────────────────────────────
    train_component = command(
        name="train",
        display_name="Step 2 — Train (7 models → best → tune)",
        description=(
            "Compare 7 classifiers with default params (CV ROC-AUC), "
            "pick winner, run RandomizedSearchCV on winner, "
            "retrain on full set, save model + metadata."
        ),
        command=(
            "python src/train.py"
            " --input_train ${{inputs.train_data}}/train.csv"
            " --output_model ${{outputs.model_dir}}"
        ),
        inputs={
            "train_data": Input(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT,
            ),
        },
        outputs={
            "model_dir": Output(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RW_MOUNT,
            ),
        },
        environment=env,
        compute=COMPUTE_CLUSTER,
        code=REPO_ROOT,        # ← params.yaml accessible here
    )

    # ── Step 3: Evaluate ───────────────────────────────────────────────────────
    evaluate_component = command(
        name="evaluate",
        display_name="Step 3 — Evaluate",
        description=(
            "Evaluate best model on held-out test set. "
            "Log accuracy, AUC, F1, precision, recall. "
            "Save confusion matrix and classification report."
        ),
        command=(
            "python src/evaluate.py"
            " --input_test  ${{inputs.test_data}}/test.csv"
            " --input_model ${{inputs.model_dir}}"
            " --output_dir  ${{outputs.eval_output}}"
        ),
        inputs={
            "test_data": Input(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT,
            ),
            "model_dir": Input(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT,
            ),
        },
        outputs={
            "eval_output": Output(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RW_MOUNT,
            ),
        },
        environment=env,
        compute=COMPUTE_CLUSTER,
        code=REPO_ROOT,        # ← consistent with other steps
    )

    # ── Wire steps ─────────────────────────────────────────────────────────────
    @pipeline(
        name="dkv_credit_default_pipeline",
        description=(
            "End-to-end credit default prediction: "
            "preprocess → multi-model train → evaluate. "
            "Auto-selects best classifier by CV ROC-AUC, "
            "tunes with RandomizedSearchCV."
        ),
        display_name="DKV Credit Default — Full Pipeline",
    )
    def credit_default_pipeline(raw_data: Input):
        pre = preprocess_component(raw_data=raw_data)
        tr  = train_component(train_data=pre.outputs.train_data)
        ev  = evaluate_component(
            test_data=pre.outputs.test_data,
            model_dir=tr.outputs.model_dir,
        )
        return {
            "train_data":  pre.outputs.train_data,
            "test_data":   pre.outputs.test_data,
            "model_dir":   tr.outputs.model_dir,
            "eval_output": ev.outputs.eval_output,
        }

    pipeline_job = credit_default_pipeline(
        raw_data=Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:credit-card-default:1",
        )
    )

    pipeline_job.settings.default_compute   = COMPUTE_CLUSTER
    pipeline_job.settings.default_datastore = "workspaceblobstore"
    pipeline_job.settings.continue_on_step_failure = False

    return pipeline_job


# ── Model registration ────────────────────────────────────────────────────────

def register_model(ml_client: MLClient, job_name: str) -> None:
    """
    Register the best model. Reads model_metadata.json from
    job outputs to tag the registered model with full details.
    """
    logger.info("Registering best model from job: %s", job_name)

    # Download metadata to read which model won and its scores
    metadata = {}
    try:
        logger.info("Downloading model_metadata.json from job outputs...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            ml_client.jobs.download(
                name=job_name,
                output_name="model_dir",
                download_path=tmp_dir,
            )
            metadata_path = os.path.join(
                tmp_dir, "named-outputs", "model_dir", "model_metadata.json"
            )
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                logger.info(
                    "Best model: %s | Default AUC: %.4f | Tuned AUC: %.4f",
                    metadata.get("best_model", "unknown"),
                    metadata.get("phase1_default_cv_auc", 0.0),
                    metadata.get("phase2_tuned_cv_auc", 0.0),
                )
            else:
                logger.warning("model_metadata.json not found — registering without metadata.")
    except Exception as e:
        logger.warning("Could not download metadata: %s", str(e))

    # Build tags from metadata
    tags = {
        "best_model":           metadata.get("best_model", "unknown"),
        "dataset":              "UCI Credit Card Default",
        "task":                 "Binary Classification",
        "selection_metric":     "roc_auc",
        "phase1_default_auc":   str(metadata.get("phase1_default_cv_auc", "")),
        "phase2_tuned_auc":     str(metadata.get("phase2_tuned_cv_auc", "")),
        "auc_improvement":      str(metadata.get("auc_improvement", "")),
        "final_train_auc":      str(metadata.get("final_train_auc", "")),
        "final_train_accuracy": str(metadata.get("final_train_accuracy", "")),
        "train_rows":           "24000",
        "test_rows":            "6000",
        "pipeline_job":         job_name,
        "experiment":           EXPERIMENT_NAME,
        "models_compared":      "7 (LR, DT, RF, KNN, AdaBoost, GradientBoosting, XGBoost)",
        "tuning_method":        "RandomizedSearchCV (n_iter=20, cv=5)",
    }

    # Add best hyperparameters to tags
    best_params = metadata.get("best_params", {})
    for k, v in best_params.items():
        tags[f"best_param_{k}"] = str(v)

    model = Model(
        name=MODEL_NAME,
        path=f"azureml://jobs/{job_name}/outputs/model_dir",
        type="custom_model",
        description=(
            f"Best classifier for credit card default prediction. "
            f"Winner: {metadata.get('best_model', 'unknown')} "
            f"(auto-selected from 7 candidates by CV ROC-AUC). "
            f"Tuned with RandomizedSearchCV. "
            f"Tuned CV AUC: {metadata.get('phase2_tuned_cv_auc', 'N/A')}. "
            f"UCI Credit Card Default dataset — 30,000 records."
        ),
        tags=tags,
    )

    registered = ml_client.models.create_or_update(model)

    logger.info("=" * 55)
    logger.info("Model registered successfully! ✓")
    logger.info("Name          : %s", registered.name)
    logger.info("Version       : %s", registered.version)
    logger.info("Best model    : %s", metadata.get("best_model", "unknown"))
    logger.info("Tuned AUC     : %s", metadata.get("phase2_tuned_cv_auc", "N/A"))
    logger.info("Best params   : %s", best_params)
    logger.info(
        "View at       : https://ml.azure.com/models/%s/version/%s"
        "?wsid=/subscriptions/%s/resourceGroups/%s"
        "/providers/Microsoft.MachineLearningServices/workspaces/%s",
        registered.name,
        registered.version,
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )
    logger.info("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ml_client    = get_ml_client()
    env          = get_or_create_environment(ml_client)
    pipeline_job = build_pipeline(env)

    # Submit
    logger.info("Submitting pipeline to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name=EXPERIMENT_NAME,
    )

    logger.info("Pipeline submitted successfully ✓")
    logger.info("Job name   : %s", returned_job.name)
    logger.info("Status     : %s", returned_job.status)
    logger.info(
        "Monitor at : https://ml.azure.com/runs/%s"
        "?wsid=/subscriptions/%s/resourceGroups/%s"
        "/providers/Microsoft.MachineLearningServices/workspaces/%s",
        returned_job.name,
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )

    # Stream logs
    logger.info("Streaming logs (Ctrl+C stops streaming, job continues)...")
    try:
        ml_client.jobs.stream(returned_job.name)
    except Exception as e:
        logger.warning("Log streaming interrupted: %s", str(e))

    # Wait for terminal status
    logger.info("Checking final job status...")
    while True:
        job = ml_client.jobs.get(returned_job.name)
        if job.status in ("Completed", "Failed", "Canceled"):
            break
        logger.info("Job status: %s — waiting 15 seconds...", job.status)
        time.sleep(15)

    logger.info("Final job status: %s", job.status)

    # Register model only on success
    if job.status == "Completed":
        logger.info("Pipeline completed. Registering best model...")
        register_model(ml_client, returned_job.name)
    else:
        logger.error(
            "Pipeline did not complete (status: %s). "
            "Model registration skipped. Check logs at: "
            "https://ml.azure.com/runs/%s?wsid=/subscriptions/%s"
            "/resourceGroups/%s/providers/Microsoft.MachineLearningServices"
            "/workspaces/%s",
            job.status,
            returned_job.name,
            SUBSCRIPTION_ID,
            RESOURCE_GROUP,
            WORKSPACE_NAME,
        )


if __name__ == "__main__":
    main()