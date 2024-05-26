from dataclasses import dataclass, field
from aligned import FeatureLocation
from aligned.schemas.feature import Feature, FeatureType
from aligned.exposed_model.interface import ExposedModel
from aligned.feature_store import FeatureReference, ModelFeatureStore, RetrivalJob
import mlflow
import polars as pl


@dataclass
class MLFlowServer(ExposedModel):
    """
    Describes a model exposed through a mlflow server.

    This also assumes that the model have a signature where each column is a feature reference.
    Meaning on the format `(feature_view|model):<contract name>:<feature name>`.
    """

    host: str

    model_alias: str
    model_name: str | None

    timeout: int = field(default=30)

    model_type: str = "mlflow_server_custom"

    @property
    def exposed_at_url(self) -> str | None:
        return self.host

    @property
    def as_markdown(self) -> str:
        return f"""Using a MLFlow server at `{self.host}`.
Assumes that it is the model: `{self.model_name}` with alias: `{self.model_alias}`."""  # noqa: E501

    def get_model_version(self, model_name: str):
        from mlflow.tracking import MlflowClient

        mlflow_client = MlflowClient()
        return mlflow_client.get_model_version_by_alias(
            self.model_name or model_name, self.model_alias
        )

    def feature_refs(self) -> list[FeatureReference]:
        import json

        info = mlflow.models.get_model_info(
            f"models:/{self.model_name}@{self.model_alias}"
        )
        signature = info.signature_dict

        if not signature:
            return []

        def from_string(string: str) -> FeatureReference:
            splits = string.split(":")
            return FeatureReference(
                name=splits[-1],
                location=FeatureLocation.from_string(":".join(splits[:-1])),
                dtype=FeatureType.string(),
            )

        return [
            from_string(feature["name"]) for feature in json.loads(signature["inputs"])
        ]

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return self.feature_refs()

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        features = await self.needed_features(store)
        req = store.store.requests_for_features(features)
        return req.request_result.entities

    async def run_polars(
        self, values: RetrivalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        import polars as pl
        from httpx import AsyncClient
        from datetime import datetime, timezone

        pred_label = list(store.model.predictions_view.labels())[0]
        pred_at = store.model.predictions_view.event_timestamp
        model_version_column = store.model.predictions_view.model_version_column
        mv = None

        if model_version_column:
            mv = self.get_model_version(store.model.name)

        job = store.features_for(values)
        df = await job.to_polars()

        features = job.request_result.feature_columns

        async with AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.host}/invocations",
                json={"dataframe_records": df[features].to_dicts()},
            )
            response.raise_for_status()
            preds = response.json()["predictions"]

        if pred_at:
            df = df.with_columns(
                pl.lit(datetime.now(timezone.utc)).alias(pred_at.name),
            )

        if mv and model_version_column:
            df = df.with_columns(
                pl.lit(mv.version).alias(model_version_column.name),
            )

        return df.with_columns(
            pl.Series(name=pred_label.name, values=preds),
        )
