import os
import pandas as pd
import mlflow
from pandas.api.types import is_string_dtype

from pathlib import Path

#MODEL_DIR = "/app/model"
MODEL_DIR = Path(__file__).resolve().parent / "model" / "current"
THRESHOLD = 0.35

try:
    #model = mlflow.pyfunc.load_model(MODEL_DIR)
    model = mlflow.xgboost.load_model(str(MODEL_DIR))
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}")

try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = df.columns.str.strip()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)

    #for c, mapping in BINARY_MAP:
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (df[c].astype(str).str.strip().map(mapping).astype("Int64").fillna(0).astype(int))

    obj_cols = [c for c in df.columns if is_string_dtype(df[c])]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df



def predict(input_dict: dict) -> str:
    try:
        df = pd.DataFrame([input_dict])
        df_enc = serve_transform(df)

        probability = float(model.predict_proba(df_enc)[0, 1])
        result = int(probability >= THRESHOLD)

    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}") from e

    if result == 1:
        return "Likely to churn"

    return "Not likely to churn"