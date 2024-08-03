from collections import defaultdict
from dataclasses import field, dataclass, asdict
import polars as pl
from typing import Callable
from aligned import Directory, ContractStore
from aligned.compiler.model import ConvertableToRetrivalJob
from aligned.feature_store import ModelFeatureStore
from aligned.retrival_job import RetrivalJob, SupervisedJob
from functools import partial
from prefect.utilities.annotations import quote

from prefect import get_run_logger, task
from prefect.runtime import flow_run
from prefect.artifacts import create_link_artifact

from src.model_registry import (
    MlFlowModelRegristry,
    ModelRegristry,
    Model,
    unpack_embeddings,
)
from src.experiment_tracker import MlFlowExperimentTracker, ExperimentTracker


@dataclass
class TrainedModelMetadata:
    model_contract: str
    prefect_training_run: str
    dataset_id: str | None


@task
async def load_store() -> ContractStore:
    from src.load_store import load_store

    return await load_store()


@task
async def generate_train_test(
    model_store: ModelFeatureStore,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    dataset_dir: Directory | None,
    dataset_id: str | None = None,
    train_size: float = 0.75,
) -> tuple[SupervisedJob, SupervisedJob]:
    from sklearn.model_selection import train_test_split

    dataset = (
        model_store.with_labels()
        .features_for(entities)
        .train_test(
            train_size=train_size,
            # Modify the classic `train_test_split` function to only take the dataframe as input
            splitter_factory=lambda config: partial(
                train_test_split,
                train_size=config.left_size,  # Left is train, right is test in this case
                test_size=config.right_size,
                random_state=123,
            ),  # type: ignore
        )
    )
    if model_store.dataset_store and dataset_dir:
        if dataset_id is None:
            dataset_id = flow_run.get_name()

        dataset = await dataset.store_dataset_at_directory(
            dataset_dir, dataset_store=model_store.dataset_store, id=dataset_id
        )

    return dataset.train, dataset.test


@task
async def generate_train_test_validate(
    model_store: ModelFeatureStore,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    dataset_dir: Directory | None,
    train_size: float = 0.6,
    test_size: float = 0.2,
    dataset_id: str | None = None,
) -> tuple[SupervisedJob, SupervisedJob, SupervisedJob]:
    from sklearn.model_selection import train_test_split

    dataset = (
        model_store.with_labels()
        .features_for(entities)
        .train_test_validate(
            train_size=train_size,
            validate_size=1 - test_size - train_size,
            # Modify the classic `train_test_split` function to only take the dataframe as input
            splitter_factory=lambda config: partial(
                train_test_split,
                train_size=config.left_size,
                test_size=config.right_size,
                random_state=123,
            ),  # type: ignore
        )
    )
    if model_store.dataset_store and dataset_dir:
        if dataset_id is None:
            dataset_id = flow_run.get_name()

        dataset = await dataset.store_dataset_at_directory(
            dataset_dir, dataset_store=model_store.dataset_store, id=dataset_id
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
        unpack_embeddings(
            dataset.input, list(data.request_result.features)
        ).to_pandas(),
        dataset.labels.to_pandas(),
    )
    return grid_search.best_params_


@task
async def fit_model(model: Model, train_set: SupervisedJob) -> Model:
    data = await train_set.to_polars()
    model.fit(
        unpack_embeddings(data.input, list(train_set.request_result.features)),
        data.labels,
    )
    return model


@task
async def set_model_as_challenger(model_id, registry: ModelRegristry) -> None:
    registry.set_model_as_alias(model_id, "challenger")


@task
async def store_model(
    model: Model, model_contract: str, store: ContractStore, registry: ModelRegristry
) -> str:
    run_name = flow_run.get_name()
    assert run_name

    metadata = TrainedModelMetadata(
        model_contract=model_contract, prefect_training_run=run_name, dataset_id=None
    )

    dataset_store = store.model(model_contract).dataset_store
    if dataset_store:
        dataset = await dataset_store.metadata_for(run_name)
        if dataset:
            metadata.dataset_id = run_name

    return registry.store_model(model, model_contract, store, metadata=asdict(metadata))


@dataclass
class RelativeThreshold:
    metric_name: str
    absolute_threshold: float | None = field(default=None)
    percentage_threshold: float | None = field(default=None)


@dataclass
class BaselineEvaluation:
    models: list[Model]
    thresholds: list[RelativeThreshold]


@dataclass
class Metric:
    method: Callable[[pl.Series, pl.Series], float]
    is_greater_better: bool


@task
async def evaluate_against_baseline_model(
    baseline: Model,
    eval_set: SupervisedJob,
    model_scores: dict[str, float],
    metrics: dict[str, Metric],
    thresholds: list[RelativeThreshold],
) -> None:
    import inspect

    logger = get_run_logger()

    eval = await eval_set.to_polars()

    if inspect.iscoroutinefunction(baseline.predict):
        baseline_preds = await baseline.predict(eval.input)
    else:
        baseline_preds = baseline.predict(eval.input)

    for threshold in thresholds:
        metric = metrics[threshold.metric_name]
        score = metric.method(eval.labels, baseline_preds)
        compare = model_scores[threshold.metric_name]
        improvement = compare - score

        logger.info(f"Improvement on {threshold.metric_name} -> {improvement}")

        if threshold.absolute_threshold:
            if metric.is_greater_better:
                assert (
                    improvement > threshold.absolute_threshold
                ), f"Failed to score higher than {threshold.absolute_threshold} on '{threshold.metric_name}'. Scored {improvement}"
            else:
                assert (
                    improvement < threshold.absolute_threshold
                ), f"Failed to score lower than {threshold.absolute_threshold} on '{threshold.metric_name}'. Scored {improvement}"


@task
async def evaluate_model_against_baseline(
    baseline: BaselineEvaluation,
    train_dataset: SupervisedJob,
    eval_sets: list[tuple[str, SupervisedJob]],
    scores: dict[str, dict[str, float]],
    metrics: dict[str, Metric],
):
    train = await train_dataset.to_polars()

    for baseline_model in baseline.models:
        baseline_model.fit(train.input, train.labels)

        for dataset_name, dataset in eval_sets:
            await evaluate_against_baseline_model(
                baseline_model,
                quote(dataset),
                scores[dataset_name],
                metrics,
                baseline.thresholds,
            )


@task
async def evaluate_model(
    model_id: str,
    datasets: list[tuple[str, SupervisedJob]],
    model_registry: ModelRegristry,
    tracker: ExperimentTracker,
    metrics: dict[str, Metric],
    thresholds: dict[str, float],
) -> dict[str, dict[str, float]]:
    from sklearn.metrics import (
        RocCurveDisplay,
        ConfusionMatrixDisplay,
        DetCurveDisplay,
        PrecisionRecallDisplay,
    )

    logger = get_run_logger()
    logger.info(f"Model ID: {model_id}")

    report_url = tracker.report_url()
    if report_url:
        _ = await create_link_artifact(
            report_url,
            "Model Report",
            description="Link to the model report containing evaluation metrics and figures.",
            key="report-url",
        )

    model = model_registry.load_model_with_id(model_id)
    if not model:
        raise ValueError(
            "Model not found in registry, something may be wrong with the storage."
        )

    preds_figures = [
        ("confusion_matrix", ConfusionMatrixDisplay.from_predictions),
        ("precision_recall_curve", PrecisionRecallDisplay.from_predictions),
    ]
    estimator_figures = [
        ("roc_curve", RocCurveDisplay.from_estimator),
        ("det_curve", DetCurveDisplay.from_estimator),
    ]

    scores: dict[str, dict[str, float]] = defaultdict(dict)

    for dataset_name, dataset in datasets:
        logger.info(f"Evaluating dataset: {dataset_name}")
        data = await dataset.to_polars()
        tracker.log_metric(f"{dataset_name}_size", data.data.height)

        unpacked = unpack_embeddings(data.input, list(dataset.request_result.features))
        preds = model.predict(unpacked)

        for scorer_name, metric in metrics.items():
            score = metric.method(data.labels, preds)
            scores[dataset_name][scorer_name] = score

            logger.info(f"Score '{scorer_name}' for '{dataset_name}': {score}")
            tracker.log_metric(f"{dataset_name}_{scorer_name}", score)

            if scorer_name in thresholds:
                value = thresholds[scorer_name]
                if metric.is_greater_better:
                    assert (
                        score > value
                    ), f"Failed to score higher than {value} on '{scorer_name}'. Scored {score}"
                else:
                    assert (
                        score < value
                    ), f"Failed to score lower than {value} on '{scorer_name}'. Scored {score}"

        for figure_name, figure in preds_figures:
            logger.info(f"Figure: {figure_name}")
            display = figure(preds, data.labels).figure_
            tracker.log_figure(display, f"{dataset_name}_{figure_name}")

        for figure_name, figure in estimator_figures:
            logger.info(f"Figure: {figure_name}")
            display = figure(model, unpacked, data.labels).figure_
            tracker.log_figure(display, f"{dataset_name}_{figure_name}")

    return scores


def default_metrics():
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    return {
        "accuracy": Metric(accuracy_score, is_greater_better=True),
        "precision": Metric(precision_score, is_greater_better=True),
        "recall": Metric(recall_score, is_greater_better=True),
        "f1": Metric(f1_score, is_greater_better=True),
        "roc_auc": Metric(roc_auc_score, is_greater_better=True),
    }


async def classifier_from_train_test_set(
    store: ContractStore,
    model_contract: str,
    entities: ConvertableToRetrivalJob | RetrivalJob,
    model: Model,
    dataset_dir: Directory | None = None,
    dataset_id: str | None = None,
    train_size: float = 0.6,
    model_contract_version: str | None = None,
    param_search: dict | None = None,
    registry: ModelRegristry | None = None,
    tracker: ExperimentTracker | None = None,
    metrics: dict[str, Metric] | None = None,
    metric_creteria: dict[str, float] | None = None,
    baseline_creteria: BaselineEvaluation | None = None,
):
    if not registry:
        registry = MlFlowModelRegristry()

    if not tracker:
        tracker = MlFlowExperimentTracker()

    if not metrics:
        metrics = default_metrics()

    if metric_creteria or baseline_creteria:
        metric_names = set((metric_creteria or dict()).keys())
        if baseline_creteria:
            metric_names.update(
                [threshold.metric_name for threshold in baseline_creteria.thresholds]
            )

        for metric_name in metric_names:
            assert (
                metric_name in metrics
            ), f"Threshold for {metric_name} is impossible as it is not part of the computed metrics."

    model_store = store.model(model_contract)

    if model_contract_version:
        model_store = model_store.using_version(model_contract_version)

    train, test = await generate_train_test(
        model_store,
        entities,
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        train_size=train_size,
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
        model = await fit_model(model, quote(train))

        model_id = await store_model(model, model_contract, store, registry)
        scores = await evaluate_model(
            model_id,
            quote(datasets),
            registry,
            tracker,
            metrics=metrics,
            thresholds=metric_creteria or {},
        )

        if baseline_creteria:
            await evaluate_model_against_baseline(
                baseline=baseline_creteria,
                train_dataset=quote(train),
                eval_sets=quote(datasets[1:]),
                scores=scores,
                metrics=metrics,
            )

        await set_model_as_challenger(model_id, registry)
