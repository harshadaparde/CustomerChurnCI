
from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain a Churn column")

    df = df.drop(columns=["customerID"], errors="ignore")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    df = df.dropna(subset=["Churn"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Churn must contain Yes or No")

    return X, y.astype(int)
