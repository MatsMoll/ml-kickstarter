from typing import Any, Protocol
from aligned import FeatureLocation, FeatureStore
from aligned.schemas.feature import Feature

import mlflow
from mlflow.models import ModelSignature
import polars as pl
import numpy as np


class Model(Protocol):
    def fit(self, X: pl.DataFrame, y: pl.Series): ...

    def predict(self, X: pl.DataFrame) -> pl.Series: ...

    def get_params(self) -> dict: ...

    def set_params(self, **kwargs) -> None: ...


def unpack_embeddings(input: pl.DataFrame, features: list[Feature]) -> pl.DataFrame:
    """
    Unpacks all embedding features in the input DataFrame.

    This is needed as most models do not accept nested data structures.
    Therefore, this will transform the input to one 1D vector.
    """

    list_features = [feature.name for feature in features if feature.dtype.is_embedding]
    if not list_features:
        return input

    input = input.with_columns(
        [
            pl.col(feature).list.to_struct(n_field_strategy="max_width")
            for feature in list_features
        ]
    )
    input = input.unnest(list_features)

    return input.select(pl.all().exclude(list_features))


def unpack_lists(input: pl.DataFrame) -> pl.DataFrame:
    """
    Unpacks all list features in the input DataFrame.

    This is needed as most models do not accept nested data structures.
    Therefore, this will transform the input to one 1D vector.
    """

    if isinstance(input, dict):
        input = pl.DataFrame(input)

    list_features = [
        feature_name
        for feature_name, datatype in input.schema.items()
        if datatype == pl.List
    ]

    input = input.with_columns(
        [
            pl.col(feature).list.to_struct(n_field_strategy="max_width")
            for feature in list_features
        ]
    )
    input = input.unnest(list_features)

    return input.select(pl.all().exclude(list_features))


class ModelRegristry(Protocol):
    def store_model(
        self,
        model: Model,
        model_name: str,
        store: FeatureStore,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError(type(self))

    def load_model_with_id(self, id: str) -> Model | None:
        raise NotImplementedError(type(self))

    def load_model(self, name: str, alias: str) -> Model | None:
        raise NotImplementedError(type(self))

    def id_for_alias(self, alias: str, model_name: str) -> str:
        raise NotImplementedError(type(self))

    def set_model_as_alias(self, model_id: str, state: str) -> None:
        raise NotImplementedError(type(self))

    def metadata_for_id(self, model_id: str) -> dict[str, Any]:
        raise NotImplementedError(type(self))


class InMemoryModelRegristry(ModelRegristry):
    models: dict[str, Model]

    def __init__(self, models: dict[str, Model] | None = None):
        self.models = models or {}

    def store_model(
        self,
        model: Model,
        model_name: str,
        store: FeatureStore,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        from uuid import uuid4

        id = str(uuid4())
        self.models[id] = model
        return id

    def load_model_with_id(self, id: str) -> Model | None:
        return self.models.get(id)

    def load_model(self, name: str) -> Model | None:
        for id, model in self.models.items():
            if id == name:
                return model

        return None


def signature_for_model(
    model_contract_name: str, store: FeatureStore
) -> ModelSignature:
    from mlflow.types.schema import ColSpec, Schema, TensorSpec
    from aligned.schemas.feature import Feature

    model_contract = store.model(model_contract_name)

    input_req = model_contract.request()
    input_reqs = input_req.needed_requests

    output_schema = None
    pred_view = model_contract.model.predictions_view

    try:
        labels = pred_view.labels()
    except Exception:
        labels = None

    def mlflow_spec(feature: Feature, location: FeatureLocation):
        dtype = feature.dtype

        ref_id = feature.as_reference(location).identifier

        if dtype.name == "float":
            return ColSpec("float", name=ref_id)
        elif dtype.name == "double":
            return ColSpec("double", name=ref_id)
        elif dtype.name == "string":
            return ColSpec("string", name=ref_id)
        elif dtype.is_numeric:
            return ColSpec("integer", name=ref_id)
        elif dtype.is_datetime:
            return ColSpec("datetime", name=ref_id)
        elif dtype.is_embedding:
            return TensorSpec(
                type=np.dtype(np.float32),
                shape=(-1, dtype.embedding_size()),
                name=ref_id,
            )
        return dtype.name

    if labels:
        output_schema = Schema(
            [
                mlflow_spec(label, FeatureLocation.model(model_contract_name))  # type: ignore
                for label in labels
            ]
        )

    all_features = []
    for request in input_reqs:
        for feature in sorted(request.returned_features, key=lambda feat: feat.name):
            if feature.name not in input_req.features_to_include:
                continue

            all_features.append(mlflow_spec(feature, request.location))

    return ModelSignature(
        inputs=Schema(all_features),
        outputs=output_schema,  # type: ignore
    )


class MlFlowModelRegristry(ModelRegristry):
    def store_model(
        self,
        model: Model,
        model_name: str,
        store: FeatureStore,
        metadata: dict[str, str] | None = None,
    ) -> str:
        from sklearn.base import BaseEstimator

        client = mlflow.client.MlflowClient()
        if isinstance(model, BaseEstimator):
            info = mlflow.sklearn.log_model(
                model, model_name, signature=signature_for_model(model_name, store)
            )
        else:
            raise ValueError("Only sklearn models are supported")

        try:
            version = mlflow.register_model(
                info.model_uri, name=model_name, tags=metadata
            )
        except Exception:
            version = client.create_model_version(
                name=model_name, source=info.model_uri, tags=metadata
            )

        try:
            mlflow.models.get_model_info(f"models:/{model_name}@champion")
        except Exception:
            client.set_registered_model_alias(
                model_name, "champion", version=version.version
            )

        return f"models:/{model_name}/{version.version}"

    def set_model_as_alias(self, model_id: str, state: str) -> None:
        client = mlflow.client.MlflowClient()
        components = model_id.split("@")[0].split("/")

        version = components[-1]
        model_name = components[-2]

        client.set_registered_model_alias(model_name, state, version)

    def metadata_for_id(self, model_id: str) -> dict[str, Any]:
        client = mlflow.client.MlflowClient()
        version = client.get_model_version_by_alias(model_id, "")
        return version.tags

    def load_model_with_id(self, id: str) -> Model | None:
        return mlflow.sklearn.load_model(id)

    def id_for_alias(self, alias: str, model_name: str) -> str:
        client = mlflow.client.MlflowClient()
        version = client.get_model_version_by_alias(model_name, alias)
        return f"models:/{model_name}/{version.version}"

    def load_model(self, name: str, alias: str = "champion") -> Model | None:
        uri = f"models:/{name}@{alias}"
        return mlflow.sklearn.load_model(uri)
