import argparse
import logging
import os
import sys

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Validate raw dataset columns.")
    parser.add_argument("--input_data",    type=str, default=None)
    parser.add_argument("--output_status", type=str, default=None)
    args = parser.parse_args()

    # Fall back to local paths when not running inside Azure ML
    if args.input_data is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.input_data    = os.path.join(BASE_DIR, "data", "credit_card_default_raw.xls")
        args.output_status = os.path.join(BASE_DIR, "artifacts", "validation_status.txt")
    return args


def load_schema() -> dict:
    """Load schema.yaml from repo root."""
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema_path = os.path.join(base_dir, "schema.yaml")

    if not os.path.exists(schema_path):
        logger.error("schema.yaml not found at: %s", schema_path)
        sys.exit(1)

    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)

    logger.info("Schema loaded from: %s", schema_path)
    return schema


def load_data(input_path: str) -> pd.DataFrame:
    """Load raw XLS file. Handle both direct file path and folder (Azure ML)."""
    logger.info("Loading data from: %s", input_path)

    # Azure ML passes a folder — find the XLS file inside
    if os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            if fname.endswith(".xls") or fname.endswith(".xlsx"):
                input_path = os.path.join(input_path, fname)
                break

    # Row 0 is a title row — actual headers are on row 1
    df = pd.read_excel(input_path, engine="xlrd", header=1)

    # Drop ID column — not a feature
    df.drop(columns=["ID"], errors="ignore", inplace=True)

    logger.info("Loaded %d rows x %d columns", *df.shape)
    logger.info("Columns found: %s", list(df.columns))
    return df


def validate_columns(df: pd.DataFrame, schema: dict) -> bool:
    """
    Check that all expected columns are present
    and no unexpected columns exist.
    Returns True if valid, False if not.
    """
    expected_cols = list(schema["COLUMNS"].keys())
    actual_cols   = list(df.columns)
    validation_status = True

    logger.info("=" * 55)
    logger.info("Running column validation...")
    logger.info("Expected columns : %d", len(expected_cols))
    logger.info("Actual columns   : %d", len(actual_cols))
    logger.info("=" * 55)

    # Check for missing columns
    missing_cols = [c for c in expected_cols if c not in actual_cols]
    if missing_cols:
        logger.error("FAIL — Missing columns: %s", missing_cols)
        validation_status = False
    else:
        logger.info("PASS — All expected columns are present ✓")

    # Check for unexpected columns
    extra_cols = [c for c in actual_cols if c not in expected_cols]
    if extra_cols:
        logger.error("FAIL — Unexpected columns found: %s", extra_cols)
        validation_status = False
    else:
        logger.info("PASS — No unexpected columns found ✓")

    return validation_status


def save_status(output_path: str, validation_status: bool) -> None:
    """Save validation result to status.txt — same approach as mentor project."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"Validation status: {validation_status}")

    logger.info("Status saved → %s", output_path)


def main():
    args   = get_args()
    schema = load_schema()
    df     = load_data(args.input_data)

    validation_status = validate_columns(df, schema)

    save_status(args.output_status, validation_status)

    logger.info("=" * 55)
    if validation_status:
        logger.info("DATA VALIDATION PASSED ✓ — Pipeline will proceed.")
        logger.info("=" * 55)
    else:
        logger.error("DATA VALIDATION FAILED ✗ — Pipeline will stop.")
        logger.error("=" * 55)
        sys.exit(1)  # Stops the Azure ML pipeline immediately


if __name__ == "__main__":
    main()