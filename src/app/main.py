from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict, predict_details
from src.serving.visualizations import create_risk_gauge, create_shap_chart
import base64
from pathlib import Path

FAVICON_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "favicon.png"
)

favicon_base64 = base64.b64encode(
    FAVICON_PATH.read_bytes()
).decode("utf-8")

GRADIO_HEAD = f"""
<link
    rel="icon"
    type="image/png"
    href="data:image/png;base64,{favicon_base64}"
>
"""

try:
    newspaper_theme = gr.Theme.from_hub("YTheme/TehnoX")
except Exception as error:
    print(f"Could not load Newspaper theme: {error}")
    newspaper_theme = gr.themes.Soft()


app = FastAPI(
    title = "Telco Customer Churn Prediction API",
    description = "ML API for predicting customer churn in telecom industry",
    version = "1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}


class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        result = predict(data.dict)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# GRADIO PREDICTION FUNCTION
# ============================================================
def gradio_interface(gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, 
                     TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges):
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    try:
        # Cette fonction devra retourner les détails de la prédiction
        result = predict_details(data)
        high_risk = result["probability"] >= result["threshold"]

        if high_risk:
            color = "#ff6366"
            background = "rgba(255, 99, 102, 0.12)"
            risk_level = "High churn risk"
        else:
            color = "#42ce7a"
            background = "rgba(66, 206, 122, 0.12)"
            risk_level = "Low churn risk"

        prediction_html = f"""
        <div style="
            width: 100%;
            padding: 14px 18px;
            box-sizing: border-box;
            border: 1px solid {color};
            border-radius: 9px;
            background: {background};
            color: {color};
        ">
            <div style="
                margin-bottom: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                opacity: 0.8;
            ">
                {risk_level}
            </div>

            <div style="
                font-size: 1.05rem;
                font-weight: 700;
            ">
                {result["prediction"]} ({result["probability"]:.1%})
            </div>
        </div>
        """

        gauge_figure = create_risk_gauge(probability=result["probability"], threshold=result["threshold"])
        shap_figure = create_shap_chart(encoded_data=result["encoded_data"])

        return (
            prediction_html,

            # Rend la jauge visible
            gr.Plot(
                value=gauge_figure,
                visible=True,
                label="Churn Probability"
            ),

            # Rend le graphique SHAP visible
            gr.Plot(
                value=shap_figure,
                visible=True,
                label="Factors Explaining the Prediction"
            )
        )

    except Exception as e:
        raise gr.Error(f"Prediction failed: {e}")

    
# GRADIO USER INTERFACE
with gr.Blocks(title="Telco Churn Predictor", fill_width=True) as demo:

    # Ligne principale :
    # espace gauche | formulaire central | espace droit
    with gr.Row():
        
        # Espace vide à gauche
        with gr.Column(scale=1, min_width=0):
            gr.HTML("")

        with gr.Column(scale=0, min_width=800):

            gr.HTML(
            """
            <section style="
                max-width: 760px;
                margin: 0 auto 32px auto;
                text-align: center;
            ">
                <p style="
                    margin: 0 0 8px 0;
                    font-size: 0.78rem;
                    font-weight: 600;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                    opacity: 0.65;
                ">
                    Educational Machine Learning Demo
                </p>

                <h1 style="
                    margin: 0 0 14px 0;
                    color: #7373E6;
                    font-size: 2.2rem;
                    line-height: 1.2;
                ">
                    Telco Customer Churn Predictor
                </h1>

                <p style="
                    margin: 0 0 10px 0;
                    font-size: 1.1rem;
                    font-weight: 600;
                ">
                    Estimate churn risk and understand the factors behind each prediction.

                <p style="
                    max-width: 650px;
                    margin: 0 auto;
                    line-height: 1.6;
                    opacity: 0.78;
                ">
                    Enter the customer's account, service and billing information.
                    The model will estimate their churn probability and identify the
                    features that influenced the prediction.
                </p>
            </section>
            """
        )

            with gr.Row(equal_height=False):

                with gr.Column(scale=1, min_width=235):
                    gr.Markdown("### Customer and Services")

                    gender_input = gr.Radio(
                        choices=["Male", "Female"],
                        value="Female",
                        label="Gender",
                        info="Customer's gender.",
                    )

                    partner_input = gr.Dropdown(
                        choices=["Yes", "No"],
                        value="No",
                        label="Partner",
                        info="Does anyone depend financially on the customer?",
                        show_label=True
                    )

                    dependents_input = gr.Dropdown(
                        choices=["Yes", "No"],
                        value="No",
                        label="Dependents",
                        info="Does the customer have dependents?",
                        show_label=True
                    )

                    tenure_input = gr.Number(
                        value=4,
                        minimum=0,
                        maximum=100,
                        label="Tenure",
                        info="Number of months with the company.",
                        show_label=True
                    )

                    phone_service_input = gr.Dropdown(
                        choices=["Yes", "No"],
                        value="Yes",
                        label="Phone Service",
                        info="Does the customer have phone service?",
                        show_label=True
                    )

                    multiple_lines_input = gr.Dropdown(
                        choices=["Yes", "No", "No phone service"],
                        value="No",
                        label="Multiple Lines",
                        info="Does the customer have multiple phone lines?",
                        show_label=True
                    )

                    internet_service_input = gr.Dropdown(
                        choices=["DSL", "Fiber optic", "No"],
                        value="Fiber optic",
                        label="Internet Service",
                        info="Type of internet service.",
                        show_label=True
                    )

                    monthly_charges_input = gr.Number(
                        value=50.0,
                        minimum=0,
                        maximum=200,
                        label="Monthly Charges ($)",
                        info="Customer's current monthly bill.",
                        show_label=True
                    )

                    total_charges_input = gr.Number(
                        value=100.0,
                        minimum=0,
                        maximum=10000,
                        label="Total Charges ($)",
                        info="Total amount billed since the customer joined.",
                        show_label=True
                    )

                # =============================================
                # RIGHT COLUMN
                # =============================================
                with gr.Column(scale=1, min_width=235):
                    gr.Markdown("### Options and Contract")

                    online_security_input = gr.Dropdown(
                        choices=["Yes", "No", "No internet service"],
                        value="Yes",
                        label="Online Security",
                        info="Does the customer have online security?",
                        show_label=True
                    )

                    online_backup_input = gr.Dropdown(
                        choices=["Yes", "No", "No internet service"],
                        value="Yes",
                        label="Online Backup",
                        info="Does the customer have online backup?",
                        show_label=True
                    )

                    device_protection_input = gr.Dropdown(
                        choices=["Yes", "No", "No internet service"],
                        value="No",
                        label="Device Protection",
                        info="Does the customer have device protection?",
                        show_label=True
                    )

                    tech_support_input = gr.Dropdown(
                        choices=["Yes", "No", "No internet service"],
                        value="No",
                        label="Tech Support",
                        info="Does the customer have technical support?",
                        show_label=True
                    )

                    streaming_tv_input = gr.Dropdown(
                        choices=["Yes", "No", "No internet service"],
                        value="No",
                        label="Streaming TV",
                        info="Does the customer use streaming TV?",
                        show_label=True
                    )

                    streaming_movies_input = gr.Dropdown(
                        choices=["Yes", "No", "No internet service"],
                        value="Yes",
                        label="Streaming Movies",
                        info="Does the customer use movie streaming?",
                        show_label=True
                    )

                    contract_input = gr.Dropdown(
                        choices=["Month-to-month", "One year", "Two year"],
                        value="Month-to-month",
                        label="Contract",
                        info="Customer's contract duration.",
                        show_label=True
                    )

                    paperless_billing_input = gr.Dropdown(
                        choices=["Yes", "No"],
                        value="Yes",
                        label="Paperless Billing",
                        info="Does the customer use paperless billing?",
                        show_label=True
                    )

                    payment_method_input = gr.Dropdown(
                        choices=[
                            "Electronic check",
                            "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)"
                        ],
                        value="Electronic check",
                        label="Payment Method",
                        info="Customer's payment method.",
                        show_label=True
                    )

            # =================================================
            # INPUT ORDER
            # =================================================
            all_inputs = [
                gender_input,
                partner_input,
                dependents_input,
                phone_service_input,
                multiple_lines_input,
                internet_service_input,
                online_security_input,
                online_backup_input,
                device_protection_input,
                tech_support_input,
                streaming_tv_input,
                streaming_movies_input,
                contract_input,
                paperless_billing_input,
                payment_method_input,
                tenure_input,
                monthly_charges_input,
                total_charges_input,
            ]

            # =================================================
            # PREDICTION BUTTON
            # =================================================
            predict_button = gr.Button("Predict churn", variant="primary")

            # =================================================
            # PREDICTION RESULT
            # =================================================
            prediction_output = gr.HTML(
                container=True,
                padding=True,
            )
        # Espace vide à droite
        with gr.Column(scale=1, min_width=0):
            gr.HTML("")

    # ========================================================
    # PREDICTION VISUALIZATIONS
    # Hidden before the first prediction
    # ========================================================
    with gr.Row(equal_height=True):
        gauge_output = gr.Plot(
            label="Churn Probability",
            visible=False,
            scale=2
        )

        shap_output = gr.Plot(
            label="Factors Explaining the Prediction",
            visible=False,
            scale=3
        )
    gr.Markdown("""</p>
                        <strong>Data notice:</strong> This educational application uses an anonymized sample dataset provided by IBM and made publicly available on 
                        <a href=https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data>Kaggle</a>.
    </p>""")
    # Le bloc Examples reste après les graphiques 
    with gr.Accordion("Example customers", open=False):
        # Champ caché utilisé uniquement comme première colonne du tableau
        risk_profile_input = gr.Textbox(
            label="Risk profile",
            visible=False,
            interactive=False,
        )

        gr.Markdown(
        """
        💡 **Tip:** Month-to-month contracts with fiber optic internet and electronic check payments tend to have higher churn rates.
        """
        )
        gr.Examples(
            examples=[
                # High churn risk
                ["High churn risk", "Male", "No", "No", "No", "No", "Fiber optic", "No", "No", "No", "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 1, 40.0, 40.0],

                # Low churn risk
                ["Low churn risk", "Female", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No", "Two year", "No", "Credit card (automatic)", 24, 30.0, 720.0]
            ],
            inputs=[risk_profile_input, *all_inputs],
        )

    gr.HTML(
        """
        <footer style="
            width: 100%;
            margin-top: 28px;
            padding: 18px 8px;
            border-top: 1px solid rgba(128, 128, 128, 0.35);
            text-align: center;
            font-size: 0.85rem;
            opacity: 0.8;
        ">
            This interactive prediction application and its machine learning model were developed by
            <a href="https://www.u-bordeaux.fr/universite/notre-strategie/nos-leviers/cma-competences-et-metiers-davenir/cap-ia"
            target="_blank" rel="noopener noreferrer"><strong>CAP IA</strong></a>.
            The source code is available on
            <a href="https://github.com/cap-ia/test_usecase_churn_telco"
            target="_blank" rel="noopener noreferrer">GitHub</a>
            and released under the
            <a href="https://opensource.org/license/mit" target="_blank" rel="noopener noreferrer">MIT License</a>.
        </footer>
        """
    )


    # ========================================================
    # PREDICTION EVENT
    # ========================================================
    predict_button.click(
        fn=gradio_interface,
        inputs=all_inputs,
        outputs=[
            prediction_output,
            gauge_output,
            shap_output
        ]
    )


app = gr.mount_gradio_app(app, demo, path="/ui", theme=newspaper_theme, head=GRADIO_HEAD)