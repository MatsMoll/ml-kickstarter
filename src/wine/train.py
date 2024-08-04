from typing import Any
from aligned import ContractStore, FileSource
from prefect import flow
import numpy as np

from src.pipelines.train import (
    BaselineEvaluation,
    RelativeThreshold,
    Metric,
    classifier_from_train_test_set,
    load_store,
)


@flow
async def train_wine_model(
    search_params: dict[str, list[Any]] | None = None,
    train_size: float = 0.6,
    test_size: float = 0.2,
    validate_size: float = 0.2,
    dataset_id: str | None = None,
) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    store: ContractStore = await load_store()

    # Train on all wines
    entities = store.feature_view("wine").select({"wine_id"}).all()
    model = RandomForestClassifier(
        n_estimators=10, random_state=np.random.RandomState(123)
    )
    dataset_dir = FileSource.directory("data/wine/datasets")
    total_size = train_size + test_size + validate_size

    await classifier_from_train_test_set(
        store=store,
        model_contract="is_high_quality_wine",
        entities=entities,
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        model=model,  # type: ignore [reportArgumentType]
        param_search=search_params,
        train_size=train_size / total_size,
        metrics={
            "accuracy": Metric(accuracy_score, is_greater_better=True),
            "precision": Metric(precision_score, is_greater_better=True),
            "recall": Metric(recall_score, is_greater_better=True),
            "f1": Metric(f1_score, is_greater_better=True),
            "roc_auc": Metric(roc_auc_score, is_greater_better=True),
        },
        metric_creteria={"precision": 0.7, "accuracy": 0.8, "recall": 0.4},
        baseline_creteria=BaselineEvaluation(
            models=[DummyClassifier(strategy="most_frequent")],
            thresholds=[
                RelativeThreshold(
                    metric_name="precision",
                    absolute_threshold=0.5,
                ),
                RelativeThreshold(
                    metric_name="recall",
                    absolute_threshold=0.1,
                ),
            ],
        ),
    )
