import os
from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from .preprocessing import RAW_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URI = "models:/telco_churn_lightgbm_woe@candidate"


@lru_cache(maxsize=1)
def load_model():
    """The FastAPI route and Gradio form share this one loaded model."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", tracking_uri))
    return mlflow.sklearn.load_model(os.getenv("TELCO_MODEL_URI", DEFAULT_URI))


def predict(input_dict: dict) -> str:
    details = predict_details(input_dict)
    return details["prediction"]


def predict_details(values: dict, explain: bool = False) -> dict:
    model = load_model()
    raw = pd.DataFrame([values], columns=RAW_COLUMNS)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").astype(float)
    probability = float(model.predict_proba(raw)[0, 1])
    is_churn = probability >= model.threshold
    prediction = "Likely to churn" if is_churn else "Not likely to churn"
    result = {
        "prediction": prediction,
        "churn_probability": probability,
        "threshold": float(model.threshold),
        "is_churn": is_churn,
    }
    if explain:
        result["top_factors"] = explain_customer(model, raw)
    return result


def explain_customer(model, raw: pd.DataFrame) -> list[dict]:
    """Return contributions to the model score (log-odds), not probabilities."""
    import shap

    encoded = model.encoded_features(raw)
    values = shap.TreeExplainer(model.pipeline_.named_steps["model"]).shap_values(encoded)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != encoded.shape:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

    display_values = model.pipeline_.named_steps["features"].transform(raw).iloc[0]
    factors = [
        {
            "feature": name,
            "value": str(display_values[name]),
            "effect": float(effect),
        }
        for name, effect in zip(encoded.columns, values[0], strict=True)
    ]
    return sorted(factors, key=lambda row: abs(row["effect"]), reverse=True)[:10]