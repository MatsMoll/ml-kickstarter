import pytest

from aligned import FeatureLocation
from aligned.data_source.batch_data_source import DummyDataSource

from prefect import flow
from prefect.testing.utilities import prefect_test_harness

from src.model_registry import InMemoryModelRegristry
from src.experiment_tracker import StdoutExperimentTracker
from src.pipelines.train import classifier_from_train_test_set
from src.load_store import load_store


async def setup_store():
    store = await load_store()

    for view in store.feature_views.keys():
        store.update_source_for(FeatureLocation.feature_view(view), DummyDataSource())

    for view in store.models.keys():
        store.update_source_for(FeatureLocation.model(view), DummyDataSource())

    return store


@pytest.mark.asyncio
async def test_generic_classifier_train_pipeline_using_prefect():
    from sklearn.ensemble import RandomForestClassifier
    from src.movie_review.contracts import MovieReviewIsNegative

    store = await setup_store()
    registry = InMemoryModelRegristry()
    tracker = StdoutExperimentTracker()
    model = RandomForestClassifier()

    @flow(name="test")
    async def test_flow():
        await classifier_from_train_test_set(
            store=store,
            model_contract=MovieReviewIsNegative.metadata.name,
            entities={
                "file": [str(i) for i in range(100)],
            },
            test_size=0.35,
            train_size=0.3,
            model=model,  # type: ignore
            registry=registry,
            tracker=tracker,
        )

    with prefect_test_harness():
        await test_flow()

    assert len(registry.models) == 1
