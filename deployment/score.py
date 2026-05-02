import os
import joblib
import pandas as pd

# Will hold the loaded model
model = None

# Same column lists as preprocess.py
BILL_COLS    = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
PAY_COLS     = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]


def init():
    """Runs ONCE — find and load the model into memory."""
    global model
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    print(f"Model directory: {model_dir}")

    # Search recursively for model.pkl
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        if "model.pkl" in files:
            model_path = os.path.join(root, "model.pkl")
            break

    if model_path is None:
        raise FileNotFoundError(
            f"model.pkl not found anywhere in {model_dir}. "
            f"Contents: {os.listdir(model_dir)}"
        )

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"Model loaded successfully")


def run(mini_batch):
    """Runs for each file Azure sends us. Returns predictions."""
    all_results = []

    for file_path in mini_batch:
        # Read the input file
        df = pd.read_csv(file_path)

        # Drop columns we don't use as features
        for col in ["default", "default payment next month", "ID"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Apply the same feature engineering as preprocess.py
        df["avg_bill_amt"]      = df[BILL_COLS].mean(axis=1)
        df["avg_pay_amt"]       = df[PAY_AMT_COLS].mean(axis=1)
        df["total_delay"]       = df[PAY_COLS].clip(lower=0).sum(axis=1)
        df["utilisation_ratio"] = df["avg_bill_amt"] / (df["LIMIT_BAL"] + 1e-6)
        df["payment_ratio"]     = df["avg_pay_amt"] / (df["avg_bill_amt"].abs() + 1e-6)

        # Get predictions
        predictions   = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        # Build results dataframe
        result = pd.DataFrame({
            "prediction":          predictions,
            "default_probability": probabilities.round(4),
            "label": ["Default" if p == 1 else "No Default" for p in predictions],
        })
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True)