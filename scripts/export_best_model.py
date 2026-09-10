import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Connexion à la base MLflow
tracking_uri = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
mlflow.set_tracking_uri(tracking_uri)

# Recherche de l’expérience
experiment = mlflow.get_experiment_by_name("Telco Churn")

if experiment is None:
    raise ValueError("MLflow experiment 'Telco Churn' not found")

# Recherche du run ayant le meilleur F1-score
best_runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.recall DESC"],
    max_results=1
)

if best_runs.empty:
    raise ValueError("No run with an recall metric was found")

best_run = best_runs.iloc[0]
best_run_id = best_run["run_id"]
best_recall = best_run["metrics.recall"]

# Recherche du modèle produit par ce run
logged_models = mlflow.search_logged_models(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"source_run_id='{best_run_id}'",
    output_format="list"
)

if not logged_models:
    raise ValueError(f"No logged model found for run {best_run_id}")

best_model = logged_models[0]
best_model_uri = f"models:/{best_model.model_id}"

print(f"Best run: {best_run_id}")
print(f"Best recall-score: {best_recall:.3f}")
print(f"Model ID: {best_model.model_id}")

# Dossier stable utilisé par Docker
destination = (
    PROJECT_ROOT
    / "src"
    / "serving"
    / "model"
    / "current"
)

# Remplacement de l’ancien bundle
if destination.exists():
    shutil.rmtree(destination)

# Téléchargement du modèle sélectionné
downloaded_model = Path(mlflow.artifacts.download_artifacts(artifact_uri=best_model_uri))

shutil.copytree(downloaded_model, destination)

# Récupération des artefacts du run correspondant
client = MlflowClient()

for artifact_name in ["feature_columns.txt", "preprocessing.pkl"]:
    artifact_path = Path(client.download_artifacts(run_id=best_run_id, path=artifact_name))
    shutil.copy2(artifact_path, destination / artifact_name)

print(f"Serving bundle exported to: {destination}")