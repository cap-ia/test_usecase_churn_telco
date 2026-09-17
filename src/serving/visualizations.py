import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb

from src.serving.inference import model

def create_risk_gauge(probability: float, threshold: float = 0.35) -> go.Figure:

    probability = float(np.clip(probability, 0, 1))
    threshold = float(np.clip(threshold, 0, 1))

    probability_percent = probability * 100
    threshold_percent = threshold * 100

    if probability >= threshold:
        color = "#ff6366"
        status = "Likely to churn"
    else:
        color = "#42ce7a"
        status = "Not likely to churn"

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0.24, 0.88]},
            value=probability_percent,
            number={
                "suffix": " %",
                "valueformat": ".1f",
                "font": {"size": 42, "color": "#f8fafc"},
            },
            title={
                "text": (
                    "Churn risk"
                    f"<br><span style='font-size:0.75em'>"
                    f"Threshold: {threshold_percent:.0f}%"
                    "</span>"
                ),
                "font": {"size": 22, "color": "#f8fafc"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "ticksuffix": " %",
                    "tickcolor": "#cbd5e1",
                },
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": "rgba(255,255,255,0.08)",
                "borderwidth": 0,
                "steps": [{"range": [0, threshold_percent], "color": "rgba(66,206,122,0.15)"},
                          {"range": [threshold_percent, 100], "color": "rgba(255,99,102,0.15)"}],

                "threshold": {
                    "value": threshold_percent,
                    "thickness": 0.8,
                    "line": {"color": "#f8fafc", "width": 4},
                },
            },
        )
    )

    # Le statut est maintenant placé sous la jauge
    figure.add_annotation(
        x=0.5,
        y=0.14,
        xref="paper",
        yref="paper",
        text=f"<b>{status}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font={"size": 17, "color": color},
    )

    figure.update_layout(
        template="plotly_dark",
        height=570,
        margin={"l": 35, "r": 35, "t": 70, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
    )
    return figure


def _format_feature_name(feature_name: str) -> str:
    """
    Rend le nom d'une variable one-hot plus facile à lire.

    Exemple :
    InternetService_Fiber optic
    devient :
    InternetService: Fiber optic
    """

    if "_" in feature_name:
        column, value = feature_name.split("_", 1)
        return f"{column}: {value}"

    return feature_name


def create_shap_chart(encoded_data: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """
    Crée un graphique local expliquant la prédiction.

    Rouge  : augmente le risque de churn.
    Vert   : diminue le risque de churn.
    """

    if encoded_data.empty:
        raise ValueError("encoded_data cannot be empty")
    
    feature_names = encoded_data.columns.tolist()

    # Récupération du Booster XGBoost
    booster = model.get_booster()

    # Format attendu par XGBoost
    data_matrix = xgb.DMatrix(encoded_data, feature_names=feature_names,)

    # Contributions SHAP natives de XGBoost
    contributions = booster.predict(data_matrix, pred_contribs=True)

    # La dernière valeur est le biais du modèle
    shap_values = contributions[0, :-1]

    shap_data = pd.DataFrame({
        "feature": feature_names,
        "contribution": shap_values,
    })

    shap_data["absolute_contribution"] = (shap_data["contribution"].abs())
    shap_data["display_name"] = (shap_data["feature"].apply(_format_feature_name))

    # Sélection des variables les plus influentes
    top_features = (shap_data.nlargest(top_n, "absolute_contribution").sort_values("absolute_contribution"))
    positive_features = top_features[top_features["contribution"] >= 0]
    negative_features = top_features[top_features["contribution"] < 0]

    figure = go.Figure()
    if not positive_features.empty:
        figure.add_trace(
            go.Bar(
                x=positive_features["contribution"],
                y=positive_features["display_name"],
                orientation="h",
                name="Increases churn risk",
                marker_color="#ff6366",
                text=[f"{value:+.3f}" for value in positive_features["contribution"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "%{y}<br>"
                    "SHAP contribution: %{x:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    if not negative_features.empty:
        figure.add_trace(
            go.Bar(
                x=negative_features["contribution"],
                y=negative_features["display_name"],
                orientation="h",
                name="Decreases churn risk",
                marker_color="#42ce7a",
                text=[f"{value:+.3f}" for value in negative_features["contribution"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "%{y}<br>"
                    "SHAP contribution: %{x:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_vline(
        x=0,
        line_width=2,
        line_color="#f8fafc",
    )

    figure.update_yaxes(
        categoryorder="array",
        categoryarray=top_features["display_name"].tolist(),
        automargin=True,
    )

    figure.update_xaxes(
        title="SHAP contribution to the model score",
        gridcolor="rgba(255,255,255,0.10)",
        zeroline=False,
    )

    figure.update_layout(
        title={"text": "Factors explaining the prediction", "x": 0.02},
        template="plotly_dark",
        barmode="relative",
        height=max(380, 55 * len(top_features) + 130),
        margin={"l": 190, "r": 70, "t": 90, "b": 65},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
    )

    return figure