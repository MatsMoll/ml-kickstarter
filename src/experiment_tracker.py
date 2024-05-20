from typing import Protocol
import mlflow
from contextlib import contextmanager
from matplotlib.figure import Figure


class ExperimentTracker(Protocol):

    @contextmanager
    def start_run(self, run_name: str):
        raise NotImplementedError(type(self))

    def log_model_params(self, params: dict) -> None:
        raise NotImplementedError(type(self))

    def log_metric(self, key: str, value: float) -> None:
        raise NotImplementedError(type(self))

    def log_figure(self, figure: Figure, name: str) -> None:
        raise NotImplementedError(type(self))

    def report_url(self) -> str | None:
        raise NotImplementedError(type(self))



class MlFlowExperimentTracker(ExperimentTracker):

    @contextmanager
    def start_run(self, run_name: str):
        mlflow.set_experiment(run_name)
        with mlflow.start_run(run_name=run_name):
            yield

    def log_model_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float) -> None:
        mlflow.log_metric(key, value)
        
    def log_figure(self, figure: Figure, name: str) -> None:
        if not name.endswith(".png"):
            name = name + ".png"

        mlflow.log_figure(figure, name)

    def report_url(self) -> str | None:
        run = mlflow.active_run()
        if not run:
            return None

        experiment_id = run.info.experiment_id
        run_id = run.info.run_id
        url = mlflow.get_tracking_uri()

        return f"{url}/#/experiments/{experiment_id}/runs/{run_id}"

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

    def report_url(self) -> str | None:
        return None

