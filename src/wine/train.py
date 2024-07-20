from aligned import ContractStore, FileSource
from prefect import flow
import numpy as np

from src.pipelines.train import classifier_from_train_test_set, load_store


@flow
async def train_wine_model(
    search_params: dict[str, list] | None = None,
    train_size: float = 0.6,
    test_size: float = 0.2,
    validate_size: float = 0.2,
    dataset_id: str | None = None,
):
    from sklearn.ensemble import RandomForestClassifier

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
        model=model,  # type: ignore
        param_search=search_params,
        train_size=train_size / total_size,
        test_size=test_size / total_size,
    )
