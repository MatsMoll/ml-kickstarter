from typing import Protocol
import mlflow
from contextlib import contextmanager


class ExperimentTracker(Protocol):

    @contextmanager
    def start_run(self, run_name: str):
        raise NotImplementedError(type(self))

    def log_model_params(self, params: dict):
        raise NotImplementedError(type(self))

    def log_metric(self, key: str, value: float):
        raise NotImplementedError(type(self))



class MlFlowExperimentTracker(ExperimentTracker):

    @contextmanager
    def start_run(self, run_name: str):
        mlflow.set_experiment(run_name)
        with mlflow.start_run(run_name=run_name):
            yield

    def log_model_params(self, params: dict):
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float):
        mlflow.log_metric(key, value)

class StdoutExperimentTracker(ExperimentTracker):

    @contextmanager
    def start_run(self, run_name: str):
        print(f"Starting run {run_name}")
        yield
        print(f"Ending run {run_name}")

    def log_model_params(self, params: dict):
        print(f"Logging params: {params}")

    def log_metric(self, key: str, value: float):
        print(f"Logging metric {key}: {value}")
