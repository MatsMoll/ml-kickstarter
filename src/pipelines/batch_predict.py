from aligned import ContractStore

from datetime import timedelta, datetime
from aligned.schemas.model import Model
from prefect import get_run_logger, flow, task


@task
async def load_store() -> ContractStore:
    from src.load_store import load_store

    return await load_store()


@task
async def batch_predict_for(model_name: str, store: ContractStore) -> None:
    logger = get_run_logger()
    model_store = store.model(model_name)
    model = model_store.model
    logger.info(f"Checking if {model.name} is up to date.")

    if not model.predictions_view.source:
        logger.info(f"No source for {model.name}. Therefore, no need to predict.")
        return

    requests = model_store.input_request().needed_requests
    if len(requests) != 1:
        logger.info(f"Needed {len(requests)} expected it would only be one.")
        return

    request = requests[0]

    last_update: datetime | None = None

    acceptable_freshness = model.predictions_view.acceptable_freshness
    if acceptable_freshness:
        last_update = await model_store.prediction_freshness()

        if last_update:
            freshness = datetime.now(tz=last_update.tzinfo) - last_update

            if freshness < acceptable_freshness:
                logger.info(
                    f"Freshness was {freshness} which is lower than {acceptable_freshness}. Therefore, skipping materialization."
                )
                return

    input_source = request.location
    logger.info(f"Predicting for `{model.name}`")

    if last_update:
        logger.info(f"Incremental update from {last_update}")

        if input_source.location == "model":
            input = (
                store.model(input_source.name)
                .predictions_between(last_update, datetime.now(tz=last_update.tzinfo))
                .select_columns(request.all_returned_columns)
            )
        else:
            input = (
                store.feature_view(input_source.name)
                .between_dates(last_update, datetime.now(tz=last_update.tzinfo))
                .select_columns(request.all_returned_columns)
            )

        await model_store.predict_over(input).upsert_into_output_source()
    else:
        logger.info("Updating everything")

        if input_source.location == "model":
            input = (
                store.model(input_source.name)
                .all_predictions()
                .select_columns(request.all_returned_columns)
            )
        else:
            input = (
                store.feature_view(input_source.name)
                .all()
                .select_columns(request.all_returned_columns)
            )

        await model_store.predict_over(input).insert_into_output_source()


intervals = {
    "minutely": (timedelta(minutes=10), timedelta(seconds=0)),
    "hourly": (timedelta(hours=1), timedelta(minutes=10)),
    "daily": (timedelta(days=1), timedelta(hours=1)),
    "weekly": (timedelta(weeks=1), timedelta(days=1)),
}


@task
def select_models_to_predict_for(
    store: ContractStore, update_tag: str | None
) -> list[Model]:
    potential_models = [
        model
        for model_name, model in store.models.items()
        if store.model(model_name).has_one_source_for_input_features()
    ]

    if update_tag is None:
        return [
            model
            for model in potential_models
            if model.predictions_view.acceptable_freshness is None
        ]

    if update_tag not in intervals:
        raise ValueError(
            f"Invalid update tag: {update_tag}. Expected {intervals.keys()}"
        )

    upper_interval, lower_interval = intervals[update_tag]

    models_to_predict_over = []
    for model in potential_models:
        limit = model.predictions_view.acceptable_freshness

        if not limit:
            continue

        if lower_interval < limit and limit <= upper_interval:
            models_to_predict_over.append(model)

    return models_to_predict_over


@flow
async def batch_predict(update_tag: str | None):
    logger = get_run_logger()
    store = await load_store()

    models_to_predict_for = select_models_to_predict_for(store, update_tag)
    if not models_to_predict_for:
        logger.info(f"Found no views for {update_tag}")
    else:
        logger.info(f"Found {len(models_to_predict_for)} views for tag: {update_tag}")
        await batch_predict_for.map(models_to_predict_for, store=store)
