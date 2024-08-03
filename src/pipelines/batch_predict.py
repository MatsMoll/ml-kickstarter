from aligned import ContractStore

from datetime import datetime
from prefect import get_run_logger, task


@task
async def load_store() -> ContractStore:
    from src.load_store import load_store

    return await load_store()


@task
async def batch_predict_for(model_name: str, store: ContractStore) -> None:
    logger = get_run_logger()
    model_store = store.model(model_name)
    model = model_store.model

    if not model.predictions_view.source:
        logger.info(f"No source for '{model.name}'. Therefore, no need to predict.")
        return

    requests = model_store.input_request().needed_requests
    if len(requests) != 1:
        logger.info(f"Needed {len(requests)} expected it would only be one.")
        return

    request = requests[0]

    last_update: datetime | None = None

    logger.info(f"Checking if '{model.name}' is up to date.")
    acceptable_freshness = model.predictions_view.acceptable_freshness
    if acceptable_freshness:
        last_update = await model_store.prediction_freshness()

        if last_update:
            freshness = datetime.now(tz=last_update.tzinfo) - last_update

            if freshness < acceptable_freshness:
                logger.info(
                    f"Freshness was {freshness} which is lower than {acceptable_freshness}. "
                    f"Therefore, skipping materialization of '{model_name}'."
                )
                return

    input_source = request.location
    logger.info(f"Predicting for '{model.name}`")

    if last_update:
        logger.info(f"Incremental update '{model.name}' from {last_update}")

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
        logger.info(f"Updating everything '{model.name}'")

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
