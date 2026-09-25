Telco churn

LightGBM model with WOE encoding, Optuna tuning, MLflow tracking, and a FastAPI/Gradio demo.

Run locally

Use Python 3.12 and install the dependencies:

python -m pip install -r requirements.txt

Place Telco-Customer-Churn.csv in data/external/ or specify its path with --data.
The local training settings are in configs/train.yaml. The decision threshold
is saved inside the trained model, so inference uses the same value.

PYTHONPATH=src python -m telco_churn.train --data data/external/Telco-Customer-Churn.csv
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src uvicorn telco_churn.app:app --reload

The API documentation is at http://127.0.0.1:8000/docs, and the Gradio demo is
at http://127.0.0.1:8000/churn-predictor_demo_cap-ia/.

Training logs the new model to MLflow and assigns the candidate alias. Serving
uses the exported model/ directory by default. To deploy a newly trained
version, export that registered version to model/, check it with the tests,
and rebuild the image. Set TELCO_MODEL_URI to another local model directory
or an accessible MLflow Registry URI to override the default.

The MLflow database, training data, and run artifacts are local files and are
not included in Git.