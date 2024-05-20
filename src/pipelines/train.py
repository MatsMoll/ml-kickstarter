from aligned import ContractStore, Directory, ContractStore
from aligned.compiler.model import ConvertableToRetrivalJob
from aligned.feature_store import ModelFeatureStore
from aligned.retrival_job import RetrivalJob, SupervisedJob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay, DetCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from functools import partial
from prefect.utilities.annotations import quote

from prefect import get_run_logger, task
from prefect.runtime import flow_run
from prefect.artifacts import create_link_artifact

from src.model_registry import MlFlowModelRegristry, AlignedModel, ModelRegristry, Model, unpack_embeddings
from src.experiment_tracker import MlFlowExperimentTracker, ExperimentTracker


@task
async def load_store() -> ContractStore:
    return await ContractStore.from_dir(".")

@task
async def generate_train_test(
    model_store: ModelFeatureStore,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    dataset_dir: Directory,
    dataset_id: str | None = None,
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
                train_size=config.left_size, # Left is train, right is test in this case
                test_size=config.right_size, 
                random_state=123
            ) # type: ignore
        )
    )
    if model_store.dataset_store:

        if dataset_id is None:
            dataset_id = flow_run.get_name()

        dataset = await dataset.store_dataset_at_directory(
            dataset_dir,
            dataset_store=model_store.dataset_store,
            id=dataset_id
        )

    return dataset.train, dataset.test


@task
async def generate_train_test_validate(
    model_store: ModelFeatureStore,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    dataset_dir: Directory | None,
    train_size: float = 0.6,
    test_size: float = 0.2,
    dataset_id: str | None = None
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
            validate_size=1 - test_size - train_size,
            # Modify the classic `train_test_split` function to only take the dataframe as input
            splitter_factory=lambda config: partial(
                train_test_split, 
                train_size=config.left_size, 
                test_size=config.right_size, 
                random_state=123
            ) # type: ignore
        )
    )
    if model_store.dataset_store and dataset_dir:

        if dataset_id is None:
            dataset_id = flow_run.get_name()

        dataset = await dataset.store_dataset_at_directory(
            dataset_dir,
            dataset_store=model_store.dataset_store,
            id=dataset_id
        )

    return dataset.train, dataset.test, dataset.validate

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
async def store_model(
    model: AlignedModel,
    store: ContractStore,
    registry: ModelRegristry
) -> str:
    return registry.store_model(model, store)

@task
async def evaluate_model(
    model_id: str,
    datasets: list[tuple[str, SupervisedJob]],
    model_registry: ModelRegristry,
    tracker: ExperimentTracker
) -> None:

    logger = get_run_logger()
    logger.info(f"Model ID: {model_id}")

    report_url = tracker.report_url()
    if report_url:
        _ = await create_link_artifact(
            report_url, 
            "Model Report", 
            description="Link to the model report containing evaluation metrics and figures.",
            key=f"report-url",
        )

    model = model_registry.load_model_with_id(model_id)
    if not model:
        raise ValueError("Model not found in registry, something may be wrong with the storage.")

    model_scorers = [
        ("accuracy", accuracy_score),
        ("precision", precision_score),
        ("recall", recall_score),
        ("f1", f1_score),
        ("roc_auc", roc_auc_score),
    ]
    preds_figures = [
        ("confusion_matrix", ConfusionMatrixDisplay.from_predictions),
        ("precision_recall_curve", PrecisionRecallDisplay.from_predictions),
    ]
    estimator_figures = [
        ("roc_curve", RocCurveDisplay.from_estimator),
        ("det_curve", DetCurveDisplay.from_estimator),
    ]

    for dataset_name, dataset in datasets:
        logger.info(f"Evaluating dataset: {dataset_name}")
        data = await dataset.to_polars()
        tracker.log_metric(f"{dataset_name}_size", data.data.height)

        unpacked = unpack_embeddings(
            data.input, list(dataset.request_result.features)
        )
        preds = model.predict(None, unpacked)

        for scorer_name, scorer in model_scorers:
            logger.info(f"Scores: {scorer_name}")
            score = scorer(data.labels, preds)
            tracker.log_metric(f"{dataset_name}_{scorer_name}", score)

        for figure_name, figure in preds_figures:
            logger.info(f"Figure: {figure_name}")
            display = figure(preds, data.labels).figure_
            tracker.log_figure(display, f"{dataset_name}_{figure_name}")

        for figure_name, figure in estimator_figures:
            logger.info(f"Figure: {figure_name}")
            display = figure(model.model, unpacked, data.labels).figure_
            tracker.log_figure(display, f"{dataset_name}_{figure_name}")



async def classifier_from_train_test_validation_set(
    store: ContractStore,
    model_contract: str,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    model: Model,
    dataset_dir: Directory | None = None,
    dataset_id: str | None = None,
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

    train, test, validate = await generate_train_test_validate(
        model_store, 
        entities, 
        dataset_dir, 
        train_size, 
        test_size,
        dataset_id=dataset_id
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
        model = await fit_model(model, quote(train), model_contract)

        model_id = await store_model(model, store, registry)
        await evaluate_model(
            model_id, 
            quote(datasets), 
            registry,
            tracker
        )



