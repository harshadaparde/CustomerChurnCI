
from pathlib import Path
import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocess import load_data


def train_model():
    X, y = load_data()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    if accuracy < 0.80:
        raise ValueError(
            f"CI failed: accuracy is {accuracy * 100:.2f}%, "
            "which is below 80%"
        )

    project_dir = Path(__file__).resolve().parent.parent
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "churn_model.joblib")

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "accuracy_percentage": round(float(accuracy * 100), 2)
    }

    with open(models_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nAccuracy requirement passed.")
    print("Model saved in models/churn_model.joblib")


if __name__ == "__main__":
    train_model()
