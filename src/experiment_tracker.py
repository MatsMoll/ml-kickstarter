import mlflow
from contextlib import contextmanager


class ExperimentTracker:

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
