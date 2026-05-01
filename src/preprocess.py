import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET       = "default"
PAY_COLS     = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS    = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
NUMERIC_COLS = ["LIMIT_BAL", "AGE"] + BILL_COLS + PAY_AMT_COLS + [
    "avg_bill_amt", "avg_pay_amt", "total_delay", "utilisation_ratio", "payment_ratio"
]


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess credit default dataset.")
    parser.add_argument("--input_data",   type=str, default=None)
    parser.add_argument("--output_train", type=str, default=None)
    parser.add_argument("--output_test",  type=str, default=None)
    args = parser.parse_args()

    # Fall back to local paths when not running inside Azure ML
    if args.input_data is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.input_data   = os.path.join(BASE_DIR, "data", "credit_card_default_raw.xls")
        args.output_train = os.path.join(BASE_DIR, "artifacts", "preprocessed_data", "train.csv")
        args.output_test  = os.path.join(BASE_DIR, "artifacts", "preprocessed_data", "test.csv")

    return args


def load_data(input_path: str) -> pd.DataFrame:
    logger.info("Loading dataset from: %s", input_path)

    # Azure ML passes a folder — find the XLS file inside
    if os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            if fname.endswith(".xls") or fname.endswith(".xlsx"):
                input_path = os.path.join(input_path, fname)
                break

    # Row 0 is a title row — actual headers are on row 1
    df = pd.read_excel(input_path, engine="xlrd", header=1)
    df.drop(columns=["ID"], inplace=True)
    df.rename(columns={"default payment next month": TARGET}, inplace=True)

    logger.info("Loaded %d rows x %d columns", *df.shape)
    logger.info("Default rate: %.2f%%", df[TARGET].mean() * 100)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    df["LIMIT_BAL"] = df["LIMIT_BAL"].clip(lower=0)
    df["AGE"]       = df["AGE"].clip(lower=18, upper=100)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})
    logger.info("Missing values after cleaning: %d", df.isnull().sum().sum())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features...")
    df["avg_bill_amt"]      = df[BILL_COLS].mean(axis=1)
    df["avg_pay_amt"]       = df[PAY_AMT_COLS].mean(axis=1)
    df["total_delay"]       = df[PAY_COLS].clip(lower=0).sum(axis=1)
    df["utilisation_ratio"] = df["avg_bill_amt"] / (df["LIMIT_BAL"] + 1e-6)
    df["payment_ratio"]     = df["avg_pay_amt"] / (df["avg_bill_amt"].abs() + 1e-6)
    logger.info("Feature engineering done. Shape: %d x %d", *df.shape)
    return df


def split_and_scale(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # Fit scaler on train only — prevents data leakage
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS]  = scaler.transform(X_test[NUMERIC_COLS])

    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    test_df  = X_test.copy()
    test_df[TARGET]  = y_test.values

    return train_df, test_df


def main():
    args = get_args()

    os.makedirs(os.path.dirname(args.output_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_test),  exist_ok=True)

    df = load_data(args.input_data)
    df = clean_data(df)
    df = engineer_features(df)

    train_df, test_df = split_and_scale(df)

    train_df.to_csv(args.output_train, index=False)
    test_df.to_csv(args.output_test,   index=False)

    logger.info("Saved train → %s", args.output_train)
    logger.info("Saved test  → %s", args.output_test)
    logger.info("Preprocessing complete ✓")


if __name__ == "__main__":
    main()