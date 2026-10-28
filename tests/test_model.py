
import pandas as pd
import pytest

from src.preprocess import load_data


def test_load_data_returns_features_and_target(tmp_path):
    df = pd.DataFrame({
        "customerID": ["C1", "C2", "C3", "C4"],
        "tenure": [1, 12, 24, 36],
        "TotalCharges": ["29.8", "", "500", "900"],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "One year"
        ],
        "Churn": ["Yes", "No", "No", "Yes"]
    })

    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)

    X, y = load_data(file_path)

    assert len(X) == 4
    assert len(y) == 4
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert X["TotalCharges"].isna().sum() == 1
    assert set(y.unique()) == {0, 1}


def test_missing_churn_column_raises_error(tmp_path):
    df = pd.DataFrame({"tenure": [1, 2]})
    file_path = tmp_path / "invalid.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError):
        load_data(file_path)
