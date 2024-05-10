import time
import asyncio
from contextlib import contextmanager
from typing import Protocol
from aligned import ContractStore, Directory, FeatureStore, FileSource
from aligned.compiler.model import ConvertableToRetrivalJob
from aligned.feature_store import ModelFeatureStore
from aligned.request.retrival_request import RequestResult
from aligned.retrival_job import RetrivalJob, SupervisedDataSet, SupervisedJob, TrainTestValidateJob
from aligned.schemas.feature import Feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from functools import partial
import numpy as np
import polars as pl
import logging

from prefect import flow, task

from src.model_registry import MlFlowModelRegristry, AlignedModel
from src.experiment_tracker import MlFlowExperimentTracker

logger = logging.getLogger(__name__)


def unpack_embeddings(input: pl.DataFrame, features: list[Feature]) -> pl.DataFrame:

    list_features = [
        feature.name
        for feature in features
        if feature.dtype.is_embedding
    ]
    input = input.with_columns([
        pl.col(feature).list.to_struct(n_field_strategy="max_width")
        for feature in list_features
    ])
    input = input.unnest(list_features)

    return input.select(pl.all().exclude(list_features))


class Model(Protocol):

    def fit(self, X: pl.DataFrame, y: pl.Series):
        ...

    def predict(self, X: pl.DataFrame):
        ...

    def get_params(self) -> dict:
        ...

@task
async def generate_train_test(
    model_store: ModelFeatureStore,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    dataset_dir: Directory,
    train_size: float = 0.75,
) -> tuple[SupervisedJob, SupervisedJob]:

    dataset = (model_store
        .with_labels()
        .features_for(entities)
        .train_test(
            train_size=train_size,
            # Modify the classic `train_test_split` function to only take the dataframe as input
            splitter_factory=lambda config: partial(
                train_test_split, 
                train_size=config.left_size, 
                test_size=config.right_size, 
                random_state=123
            ) # type: ignore
        )
    )
    if model_store.dataset_store:
        dataset = await dataset.store_dataset_at_directory(
            dataset_dir,
            dataset_store=model_store.dataset_store,
        )

    return dataset.train, dataset.test


@task
async def generate_dataset(
    model_store: ModelFeatureStore,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    dataset_dir: Directory,
    train_size: float = 0.6,
    test_size: float = 0.2,
) -> tuple[
    SupervisedJob,
    SupervisedJob,
    SupervisedJob
]:
    dataset = (model_store
        .with_labels()
        .features_for(entities)
        .train_test_validate(
            train_size=train_size,
            validate_size=test_size,
            # Modify the classic `train_test_split` function to only take the dataframe as input
            splitter_factory=lambda config: partial(
                train_test_split, 
                train_size=config.left_size, 
                test_size=config.right_size, 
                random_state=123
            ) # type: ignore
        )
    )
    if model_store.dataset_store:
        dataset = await dataset.store_dataset_at_directory(
            dataset_dir,
            dataset_store=model_store.dataset_store,
            id="movie_review"
        )

    return dataset.train, dataset.test, dataset.validate

@task
async def fit_model(
    model: Model, 
    train_set: SupervisedJob,
    model_contract: str,
) -> Model:
    data = await train_set.to_polars()
    model.fit(
        unpack_embeddings(data.input, list(train_set.request_result.features)), 
        data.labels
    )
    aligned_model = AlignedModel(model, model_contract, None)
    return aligned_model

@task
def store_model(
    model: AlignedModel,
    store: FeatureStore,
    registry: MlFlowModelRegristry
) -> str:
    return registry.store_model(model, store)

@task
async def evaluate_model(
    model_id: str,
    datasets: list[tuple[str, SupervisedJob]],
    model_registry: MlFlowModelRegristry,
    tracker: MlFlowExperimentTracker
) -> None:

    model = model_registry.load_model_with_id(model_id)
    model_scorers = [
        ("accuracy", accuracy_score),
        ("precision", precision_score),
        ("recall", recall_score),
        ("f1", f1_score),
        ("roc_auc", roc_auc_score),
    ]

    for dataset_name, dataset in datasets:
        data = await dataset.to_polars()
        tracker.log_metric(f"{dataset_name}_size", data.data.height)

        unpacked = unpack_embeddings(
            data.input, list(dataset.request_result.features)
        )
        preds = model.predict(None, unpacked)

        for scorer_name, scorer in model_scorers:
            score = scorer(data.labels, preds)
            tracker.log_metric(f"{dataset_name}_{scorer_name}", score)

@task
async def find_best_parameters(
    model: Model,
    data: SupervisedJob,
    params: dict,
) -> dict:
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    grid_search = GridSearchCV(
        model,
        param_grid=params,
        cv=cross_validation,
    )

    dataset = await data.to_polars()

    grid_search.fit(
        unpack_embeddings(dataset.input, list(data.request_result.features)).to_pandas(),
        dataset.labels.to_pandas()
    )
    return grid_search.best_params_

async def classification_pipeline_train_test(
    store: FeatureStore,
    model_contract: str,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    dataset_dir: Directory,
    model: Model,
    train_size: float = 0.75,
    model_contract_version: str | None = None,
    param_search: dict | None = None
):
    registry = MlFlowModelRegristry()
    tracker = MlFlowExperimentTracker()
    model_store = store.model(model_contract)

    if model_contract_version:
        model_store = model_store.using_version(model_contract_version)

    train, test = await generate_train_test(
        model_store, 
        entities, 
        dataset_dir, 
        train_size, 
    )

    datasets = [
        ("train", train),
        ("test", test),
    ]

    with tracker.start_run(run_name=model_contract):
        if param_search:
            best_params = await find_best_parameters(model, train, param_search)
            model.set_params(**best_params)

        tracker.log_model_params(model.get_params())
        model = await fit_model(model, train, model_contract)

        model_id = store_model(model, store, registry)

        await evaluate_model(
            model_id, 
            datasets, 
            registry,
            tracker
        )

async def generic_classifier_train_pipeline_tasks_with_search(
    store: FeatureStore,
    model_contract: str,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    dataset_dir: Directory,
    model: Model,
    train_size: float = 0.6,
    test_size: float = 0.2,
    model_contract_version: str | None = None,
    param_search: dict | None = None
):
    registry = MlFlowModelRegristry()
    tracker = MlFlowExperimentTracker()
    model_store = store.model(model_contract)

    if model_contract_version:
        model_store = model_store.using_version(model_contract_version)

    train, test, validate = await generate_dataset(
        model_store, 
        entities, 
        dataset_dir, 
        train_size, 
        test_size
    )

    datasets = [
        ("train", train),
        ("test", test),
        ("validate", validate)
    ]

    with tracker.start_run(run_name=model_contract):
        if param_search:
            best_params = await find_best_parameters(model, train, param_search)
            model.set_params(**best_params)

        tracker.log_model_params(model.get_params())
        model = await fit_model(model, train, model_contract)

        model_id = store_model(model, store, registry)

        await evaluate_model(
            model_id, 
            datasets, 
            registry,
            tracker
        )


async def generic_classifier_train_pipeline_tasks(
    store: FeatureStore,
    model_contract: str,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    dataset_dir: Directory,
    model: Model,
    train_size: float = 0.6,
    test_size: float = 0.2,
    model_contract_version: str | None = None,
):
    registry = MlFlowModelRegristry()
    tracker = MlFlowExperimentTracker()
    model_store = store.model(model_contract)

    if model_contract_version:
        model_store = model_store.using_version(model_contract_version)

    train, test, validate = await generate_dataset(
        model_store, 
        entities, 
        dataset_dir, 
        train_size, 
        test_size
    )

    datasets = [
        ("train", train),
        ("test", test),
        ("validate", validate)
    ]

    if model_store.dataset_store:
        other_datasets = model_store.dataset_store.datasets_with_tag("regression")
        datasets.extend([
            (dataset.name, dataset.format_as_job(train))
            for dataset in other_datasets
        ])

    with tracker.start_run(run_name=model_contract):
        model = await fit_model(model, train, model_contract)

        model_id = store_model(model, store, registry)

        await evaluate_model(
            model_id, 
            datasets, 
            registry,
            tracker
        )



async def generic_classifier_train_pipeline(
    store: FeatureStore,
    model_contract: str,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    dataset_dir: Directory,
    model: Model,
    train_size: float = 0.6,
    test_size: float = 0.2,
    model_contract_version: str | None = None,
):
    registry = MlFlowModelRegristry()
    tracker = MlFlowExperimentTracker()

    model_scorers = [
        ("accuracy", accuracy_score),
        ("precision", precision_score),
        ("recall", recall_score),
        ("f1", f1_score),
        ("roc_auc", roc_auc_score),
    ]

    model_store = store.model(model_contract)

    if model_contract_version:
        model_store = model_store.using_version(model_contract_version)

    with tracker.start_run(run_name=model_contract):
        tracker.log_model_params(model.get_params())

        with timeit("Creating dataset"):

            dataset = (model_store
                .with_labels()
                .features_for(entities)
                .train_test_validate(
                    train_size=train_size,
                    validate_size=test_size,
                    splitter_factory=lambda config: partial(
                        train_test_split, 
                        train_size=config.left_size, 
                        test_size=config.right_size, 
                        random_state=123
                    ) # type: ignore
                )
            )
            if model_store.dataset_store:
                dataset = dataset.store_dataset_at_directory(
                    dataset_dir,
                    dataset_store=model_store.dataset_store
                )

            train_set = await dataset.train.to_polars()

        input_features = dataset.train_job.request_result.features

        with timeit("Fitting model"):
            model.fit(
                unpack_embeddings(train_set.input, list(input_features)), 
                train_set.labels
            )

        aligned_model = AlignedModel(model, model_contract, None)

        with timeit("Storing model"):
            id = registry.store_model(aligned_model, store)

        with timeit("Loading Model"):
            stored_model = registry.load_model_with_id(
                id=id, 
                name=model_contract
            )

        for dataset_name, dataset in [
            ("train", dataset.train), 
            ("test", dataset.test), 
            ("validate", dataset.validate),
        ]:
            with timeit(f"Loading dataset - {dataset_name}"):
                data = await dataset.to_polars()

            with timeit(f"Scoring dataset - {dataset_name}"):
                unpacked = unpack_embeddings(data.input, list(input_features))
                preds = stored_model.predict(None, unpacked)

                for scorer_name, scorer in model_scorers:
                    score = scorer(data.labels, preds)
                    tracker.log_metric(f"{dataset_name}_{scorer_name}", score)

                # for scorer_name, scorer in image_scorers:
                #     display = scorer(data.labels, preds)
                #     tracker.log_image(display.get_image(), f"{scorer_name}_{dataset_name}")

@task
async def load_store() -> ContractStore:
    return await ContractStore.from_dir(".")

@task
async def equal_distribution_entities(number_of_records: int, store: ContractStore) -> pl.DataFrame:
    n_per_class = number_of_records // 2

    all_entities = (await 
        store.feature_view("movie_review")
        .select_columns(["file", "is_negative"])
        .to_polars()
    )
    return all_entities.filter(pl.col("is_negative") == 1).limit(n_per_class).vstack(
        all_entities.filter(pl.col("is_negative") == 0).limit(n_per_class)
    )


@flow(name="train_sentiment")
async def train_sentiment(number_of_records: int = 1000, search_params: dict | None = None):
    print("Training sentiment")
    store = await load_store()

    entities = await equal_distribution_entities(
        number_of_records=number_of_records, 
        store=store
    )

    model = RandomForestClassifier(
        n_estimators=10, 
        random_state=np.random.RandomState(123)
    )

    await generic_classifier_train_pipeline_tasks_with_search(
        store=store,
        model_contract="movie_review_is_negative",
        entities=entities,
        dataset_dir=FileSource.directory(f"data/movie_review_is_negative/datasets"),
        model=model, # type: ignore
        param_search=search_params
        
    )

@flow(name="train_sentiment_train")
async def train_sentiment_test(
    number_of_records: int = 1000, 
    search_params: dict | None = None,
    train_size: float = 0.75,
):
    print("Training sentiment")
    store = await load_store()

    entities = await equal_distribution_entities(
        number_of_records=number_of_records, 
        store=store
    )
    model = RandomForestClassifier(
        n_estimators=10, 
        random_state=np.random.RandomState(123)
    )
    await classification_pipeline_train_test(
        store=store,
        model_contract="movie_review_is_negative",
        entities=entities,
        dataset_dir=FileSource.directory(f"data/movie_review_is_negative/datasets"),
        model=model, # type: ignore
        train_size=train_size,
        param_search=search_params
    )

