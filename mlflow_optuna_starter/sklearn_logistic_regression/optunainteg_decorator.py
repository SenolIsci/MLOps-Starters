""""
optuna.integration.MLflowCallback
Callback to track Optuna trials with MLflow.

This callback adds relevant information that is tracked by Optuna to MLflow.
track_in_mlflow(): Decorator for using MLflow logging in the objective function.

This decorator enables the extension of MLflow logging provided by the callback.

All information logged in the decorated objective function will be added to the MLflow run for the trial created by the callback.
"""

import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
mlflc = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name="my metric score",
)

@mlflc.track_in_mlflow()
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    mlflow.log_param("power", 2)
    mlflow.log_metric("base of metric", x - 2)

    return (x - 2) ** 2


study = optuna.create_study(study_name="my_other_study")
study.optimize(objective, n_trials=10, callbacks=[mlflc])


study = optuna.create_study(study_name="my_study")
study.optimize(objective, n_trials=10, callbacks=[mlflc])
