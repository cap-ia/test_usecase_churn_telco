import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
import optuna

DATA_PATH = "data/processed/Processed_Telco-Customer-Churn.csv"  # adjust to your file path
TARGET_COL = "Churn"


print("=== Phase 2: Modeling with XGBoost ===")

def main():
    df = pd.read_csv(DATA_PATH)
    print(df[TARGET_COL].dtype)
    if is_string_dtype(df[TARGET_COL]):
        df[TARGET_COL] = df[TARGET_COL].str.strip().map({"No": 0, "Yes": 1})
        print(df[TARGET_COL].unique())

    assert df[TARGET_COL].isna().sum() == 0, "Churn has NaNs"
    assert set(df[TARGET_COL].unique()) <= {0, 1}, "Churn not 0/1"

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    THRESHOLD = 0.3

    def objective(trial):
        params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= THRESHOLD).astype(int)
        return recall_score(y_test, y_pred, pos_label=1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print("Best Params:", study.best_params)
    print("Best Recall:", study.best_value)

if __name__ == "__main__":
    main()