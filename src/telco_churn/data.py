import pandas as pd
import os
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split


def load_raw_data(file_path: str) -> pd.DataFrame:
    """"""
    if not os.path.exists(file_path):
        raise FileExistsError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """"""
    df.columns = df.columns.str.strip()

    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df


def extract_target(df: pd.DataFrame, target_col: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    """"""
    if target_col not in df:
        raise ValueError(f"Missing target column: {target_col}")
    
    if target_col in df.columns and is_string_dtype(df["Churn"]):
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    y = df[target_col].astype(int).rename(target_col)
    X = df.drop(columns=target_col)

    return X, y


def make_holdout_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split before supervised encoding, imputation, or model selection."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)