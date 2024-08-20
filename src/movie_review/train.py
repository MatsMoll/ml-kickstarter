from prefect import flow, task
from aligned import ContractStore, FileSource

from src.pipelines.train import classifier_from_train_test_set, load_store

import numpy as np
import polars as pl


@task
async def equal_distribution_entities(
    number_of_records: int, store: ContractStore
) -> pl.DataFrame:
    n_per_class = number_of_records // 2

    all_entities = (
        await store.feature_view("movie_review")
        .select_columns(["review_id", "is_negative"])
        .to_polars()
    )
    return (
        all_entities.filter(pl.col("is_negative") == 1)
        .limit(n_per_class)
        .vstack(all_entities.filter(pl.col("is_negative") == 0).limit(n_per_class))
    )


@flow(name="train_movie_review_is_negative_with_train_test")
async def train_sentiment(
    number_of_records: int = 1000,
    search_params: dict | None = None,
    train_size: float = 0.75,
    dataset_id: str | None = None,
):
    from sklearn.ensemble import RandomForestClassifier

    store: ContractStore = await load_store()

    entities = await equal_distribution_entities(
        number_of_records=number_of_records, store=store
    )

    model = RandomForestClassifier(
        n_estimators=10, random_state=np.random.RandomState(123)
    )
    dataset_dir = FileSource.directory("data/movie_review_is_negative/datasets")

    await classifier_from_train_test_set(
        store=store,
        model_contract="movie_review_is_negative",
        model=model,  # type: ignore
        entities=entities,
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        param_search=search_params,
        train_size=train_size,
    )
