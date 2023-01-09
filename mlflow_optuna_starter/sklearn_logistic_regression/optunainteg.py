""""
optuna.integration.MLflowCallback
Callback to track Optuna trials with MLflow.

This callback adds relevant information that is tracked by Optuna to MLflow.
"""

import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


mlflc = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name="my metric score",
)

study = optuna.create_study(study_name="my_study")
study.optimize(objective, n_trials=10, callbacks=[mlflc])
