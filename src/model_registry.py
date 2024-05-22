from typing import Any, Protocol
from dataclasses import dataclass
from aligned import FeatureStore
from aligned.schemas.feature import Feature

import mlflow
from mlflow.models import ModelSignature
import polars as pl
import numpy as np


class Model(Protocol):

    def fit(self, X: pl.DataFrame, y: pl.Series):
        ...

    def predict(self, X: pl.DataFrame):
        ...

    def get_params(self) -> dict:
        ...

    def set_params(self, **kwargs) -> None:
        ...


def unpack_embeddings(input: pl.DataFrame, features: list[Feature]) -> pl.DataFrame:
    """
    Unpacks all embedding features in the input DataFrame.

    This is needed as most models do not accept nested data structures.
    Therefore, this will transform the input to one 1D vector.
    """

    list_features = [
        feature.name
        for feature in features
        if feature.dtype.is_embedding
    ]
    if not list_features:
        return input

    input = input.with_columns([
        pl.col(feature).list.to_struct(n_field_strategy="max_width")
        for feature in list_features
    ])
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

    input = input.with_columns([
        pl.col(feature).list.to_struct(n_field_strategy="max_width")
        for feature in list_features
    ])
    input = input.unnest(list_features)

    return input.select(pl.all().exclude(list_features))


@dataclass
class AlignedModel(mlflow.pyfunc.PythonModel):

    model: Any
    model_name: str
    model_contract_version: str | None

    def predict(self, context, input: pl.DataFrame) -> pl.Series:
        return self.model.predict(unpack_lists(input))


class ModelRegristry(Protocol):

    def store_model(self, model: AlignedModel, store: FeatureStore) -> str:
        raise NotImplementedError(type(self))

    def load_model_with_id(self, id: str) -> AlignedModel | None:
        raise NotImplementedError(type(self))

    def load_model(self, name: str) -> AlignedModel | None:
        raise NotImplementedError(type(self))



class InMemoryModelRegristry(ModelRegristry):

    models: dict[str, AlignedModel]

    def __init__(self, models: dict[str, AlignedModel] | None = None):
        self.models = models or {}

    def store_model(self, model: AlignedModel, store: FeatureStore) -> str:
        from uuid import uuid4

        id = str(uuid4())
        self.models[id] = model
        return id

    def load_model_with_id(self, id: str) -> AlignedModel | None:
        return self.models.get(id)

    def load_model(self, name: str) -> AlignedModel | None:
        for model in self.models.values():
            if model.model_name == name:
                return model

        return None

    

def signature_for_model(model: AlignedModel, store: FeatureStore) -> ModelSignature:
    from mlflow.types.schema import ColSpec, Schema, TensorSpec
    from aligned.schemas.feature import Feature

    model_contract = store.model(model.model_name)
    if model.model_contract_version:
        model_contract = model_contract.using_version(model.model_contract_version)


    input_req = model_contract.request().request_result

    output_schema = None
    pred_view = model_contract.model.predictions_view

    try:
        labels = pred_view.labels()
    except Exception:
        labels = None

    def mlflow_spec(feature: Feature):
        dtype = feature.dtype

        if dtype.name == "float":
            return ColSpec("float", name=feature.name)
        elif dtype.name == "double":
            return ColSpec("double", name=feature.name)
        elif dtype.name == "string":
            return ColSpec("string", name=feature.name)
        elif dtype.is_numeric:
            return ColSpec("integer", name=feature.name)
        elif dtype.is_datetime:
            return ColSpec("datetime", name=feature.name)
        elif dtype.is_embedding:
            return TensorSpec(type=np.dtype(np.float32), shape=(-1, dtype.embedding_size()), name=feature.name)
        return dtype.name
        

    if labels:
        output_schema = Schema([
            mlflow_spec(label) # type: ignore
            for label in labels
        ])

    return ModelSignature(
        inputs=Schema([ # type: ignore
            mlflow_spec(feature) # type: ignore
            for feature in input_req.features
        ]),
        outputs=output_schema # type: ignore
    )

class MlFlowModelRegristry(ModelRegristry):


    def store_model(self, model: AlignedModel, store: FeatureStore) -> str:
        client = mlflow.client.MlflowClient()

        info = mlflow.pyfunc.log_model(
            model.model_name, 
            python_model=model,
            signature=signature_for_model(model, store)
        )

        try:
            version = mlflow.register_model(
                info.model_uri,
                model.model_name,
                tags={
                    "model_contract_version": model.model_contract_version,
                }
            )
        except Exception:
            version = client.create_model_version(
                name=model.model_name,
                source=info.model_uri,
                tags={
                    "model_contract_version": model.model_contract_version,
                }
            )

        try:
            mlflow.models.get_model_info(f"models:/{model.model_name}@champion")
        except Exception as e:
            client.set_registered_model_alias(
                model.model_name, 
                "champion", 
                version=version.version
            )

        return info.model_uri


    def load_model_with_id(self, id: str) -> AlignedModel | None:

        model = mlflow.pyfunc.load_model(id).unwrap_python_model()
        if isinstance(model, AlignedModel):
            return model

        return None


    def load_model(self, name: str, alias: str = "Champion") -> AlignedModel | None:
        uri = f"models:/{name}@{alias}"
        model = mlflow.pyfunc.load_model(uri).python_model
        if isinstance(model, AlignedModel):
            return model

        return None
