import os
import time
import logging
import pandas as pd

from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.entities import (
    BatchEndpoint, BatchDeployment, CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────────────────
SUBSCRIPTION_ID = "b17046df-01cb-41d8-966c-a496e8fbef0e"
RESOURCE_GROUP  = "rg-dkv-ml-casestudy"
WORKSPACE       = "dkv-ml-workspace"
COMPUTE         = "dkv-cpu-cluster"
ENVIRONMENT     = "dkv-ml-env"
MODEL_NAME      = "credit-default-best-model"
ENDPOINT_NAME   = "credit-default-batch"
DEPLOYMENT_NAME = "deploy-v1"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_DIR = os.path.join(REPO_ROOT, "deployment")
TEST_DIR  = os.path.join(REPO_ROOT, "deployment", "test_data")


# ── Connect to Azure ML ───────────────────────────────────────────────────────
def connect():
    log.info("Connecting to Azure ML...")
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE,
    )


# ── Step 1: Create endpoint ───────────────────────────────────────────────────
def create_endpoint(ml_client):
    log.info("Creating endpoint: %s", ENDPOINT_NAME)
    try:
        ml_client.batch_endpoints.get(ENDPOINT_NAME)
        log.info("Endpoint already exists — skipping.")
    except Exception:
        endpoint = BatchEndpoint(
            name=ENDPOINT_NAME,
            description="Batch endpoint for credit default prediction",
        )
        ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
        log.info("Endpoint created — waiting for it to be fully ready...")

        # Wait until endpoint is in Succeeded state
        for _ in range(20):  # max 5 minutes
            time.sleep(15)
            ep = ml_client.batch_endpoints.get(ENDPOINT_NAME)
            log.info("Endpoint state: %s", ep.provisioning_state)
            if ep.provisioning_state == "Succeeded":
                log.info("Endpoint ready ✓")
                return
        raise RuntimeError("Endpoint did not become ready in time.")


# ── Step 2: Deploy latest model to endpoint ───────────────────────────────────
def deploy_model(ml_client):
    log.info("Fetching latest version of model: %s", MODEL_NAME)
    model = ml_client.models.get(MODEL_NAME, label="latest")
    log.info("Deploying %s v%s ...", model.name, model.version)
    log.info("This takes 5–10 minutes — please wait.")

    env = ml_client.environments.get(ENVIRONMENT, label="latest")

    deployment = BatchDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model,
        environment=env,
        compute=COMPUTE,
        instance_count=1,
        mini_batch_size=10,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name="predictions.csv",
        code_configuration=CodeConfiguration(
            code=SCORE_DIR,
            scoring_script="score.py",
        ),
    )

    ml_client.batch_deployments.begin_create_or_update(deployment).result()

    # Make this the default deployment for the endpoint
    endpoint = ml_client.batch_endpoints.get(ENDPOINT_NAME)
    endpoint.defaults.deployment_name = DEPLOYMENT_NAME
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    log.info("Model deployed ✓")


# ── Step 3: Create test input file ────────────────────────────────────────────
def create_test_data():
    log.info("Creating sample test data...")
    os.makedirs(TEST_DIR, exist_ok=True)

    sample = pd.DataFrame({
        "LIMIT_BAL": [20000,  120000, 90000,  500000, 50000],
        "SEX":       [2,      2,      2,      1,      1],
        "EDUCATION": [2,      2,      2,      1,      2],
        "MARRIAGE":  [1,      2,      2,      2,      1],
        "AGE":       [24,     26,     34,     29,     57],
        "PAY_0":     [2,      -1,     0,      0,      -1],
        "PAY_2":     [2,      2,      0,      0,      0],
        "PAY_3":     [-1,     0,      0,      0,      -1],
        "PAY_4":     [-1,     0,      0,      0,      0],
        "PAY_5":     [-2,     0,      0,      0,      0],
        "PAY_6":     [-2,     2,      0,      0,      0],
        "BILL_AMT1": [3913,   2682,   29239,  367965, 8617],
        "BILL_AMT2": [3102,   1725,   14027,  412023, 5670],
        "BILL_AMT3": [689,    2682,   13559,  445007, 35835],
        "BILL_AMT4": [0,      3272,   14331,  542653, 20940],
        "BILL_AMT5": [0,      3455,   14948,  483003, 19146],
        "BILL_AMT6": [0,      3261,   15549,  473944, 19131],
        "PAY_AMT1":  [0,      0,      1518,   55000,  2000],
        "PAY_AMT2":  [689,    1000,   1500,   40000,  36681],
        "PAY_AMT3":  [0,      1000,   1000,   38000,  10000],
        "PAY_AMT4":  [0,      1000,   1000,   20239,  9000],
        "PAY_AMT5":  [0,      0,      1000,   13750,  689],
        "PAY_AMT6":  [0,      2000,   5000,   13770,  679],
    })

    csv_path = os.path.join(TEST_DIR, "sample_input.csv")
    sample.to_csv(csv_path, index=False)
    log.info("Sample data saved → %s", csv_path)


# ── Step 4: Submit test job and wait ──────────────────────────────────────────
def test_endpoint(ml_client):
    log.info("Submitting test job to endpoint...")
    job = ml_client.batch_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        input=Input(type=AssetTypes.URI_FOLDER, path=TEST_DIR),
    )
    log.info("Job submitted: %s", job.name)
    log.info("Waiting for results (3–5 minutes)...")

    while True:
        job = ml_client.jobs.get(job.name)
        log.info("Status: %s", job.status)
        if job.status == "Completed":
            break
        if job.status in ("Failed", "Canceled"):
            raise RuntimeError(f"Job failed: {job.status}")
        time.sleep(30)

    return job.name


# ── Step 5: Download predictions ──────────────────────────────────────────────
def show_results(ml_client, job_name):
    results_dir = os.path.join(REPO_ROOT, "deployment", "results")
    os.makedirs(results_dir, exist_ok=True)

    log.info("Downloading predictions...")
    ml_client.jobs.download(name=job_name, download_path=results_dir, all=True)

    # Find the predictions.csv file
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == "predictions.csv":
                path    = os.path.join(root, f)
                results = pd.read_csv(path)
                log.info("=" * 50)
                log.info("PREDICTIONS:")
                log.info("=" * 50)
                print(results.to_string(index=False))
                log.info("=" * 50)
                log.info("Saved to: %s", path)
                return

    log.warning("predictions.csv not found — check job logs.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ml_client = connect()

    log.info("─" * 50)
    log.info("STEP 1 of 4 — Create endpoint")
    log.info("─" * 50)
    create_endpoint(ml_client)

    log.info("─" * 50)
    log.info("STEP 2 of 4 — Deploy latest model (5–10 minutes)")
    log.info("─" * 50)
    deploy_model(ml_client)

    log.info("─" * 50)
    log.info("STEP 3 of 4 — Create test data")
    log.info("─" * 50)
    create_test_data()

    log.info("─" * 50)
    log.info("STEP 4 of 4 — Test the endpoint")
    log.info("─" * 50)
    job_name = test_endpoint(ml_client)

    log.info("─" * 50)
    log.info("DONE — Showing results")
    log.info("─" * 50)
    show_results(ml_client, job_name)


if __name__ == "__main__":
    main()