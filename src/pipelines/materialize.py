import asyncio
from aligned import ContractStore, FeatureLocation

from datetime import datetime
from prefect import get_run_logger, flow, task
from prefect.futures import PrefectFuture


@task
async def load_store() -> ContractStore:
    from src.load_store import load_store

    return await load_store()


@task
async def materialize_view(view_name: str, store: ContractStore) -> None:
    logger = get_run_logger()

    view_store = store.feature_view(view_name)
    view = view_store.view
    logger.info(f"Checking if {view.name} is up to date.")

    if not view.materialized_source:
        logger.info(f"No materialized source for {view.name}.")
        return

    last_update: datetime | None = None

    if view.acceptable_freshness:
        last_update = await view_store.freshness()

        logger.info(last_update)
        if last_update:
            freshness = datetime.now(tz=last_update.tzinfo) - last_update

            if freshness < view.acceptable_freshness:
                logger.info(
                    f"Freshness was {freshness} which is lower than {view.acceptable_freshness}. Therefore, skipping materialization."
                )
                return

    logger.info(f"Materialize {view.name}")
    if last_update:
        logger.info(f"Incremental update from {last_update}")
        await store.upsert_into(
            FeatureLocation.feature_view(view.name),
            view_store.using_source(view.source).between_dates(
                start_date=last_update, end_date=datetime.now(tz=last_update.tzinfo)
            ),
        )
    else:
        logger.info("Updating everything")
        await store.overwrite(
            FeatureLocation.feature_view(view.name),
            view_store.using_source(view.source).all(),
        )


async def levels_for_store(store: ContractStore) -> dict[FeatureLocation, int]:
    levels = {}

    def update_with(sub_levels: dict[FeatureLocation, int]) -> None:
        for key, value in sub_levels.items():
            set_value = max(value, levels.get(key, 0))
            levels[key] = set_value

    for view in store.feature_views.values():
        view_loc = FeatureLocation.feature_view(view.name)
        if view_loc not in levels:
            update_with(await levels_for_location(view_loc, store, levels))

    for model in store.models.values():
        model_loc = FeatureLocation.model(model.name)

        if model_loc not in levels:
            update_with(await levels_for_location(model_loc, store, levels))

    return levels


async def location_depends_on(
    location: FeatureLocation, store: ContractStore
) -> set[FeatureLocation]:
    if location.location == "model":
        model_store = store.model(location.name)
        depends_on = model_store.depends_on()
        if model_store.model.exposed_model is not None:
            depends_on.update(await model_store.model.exposed_model.depends_on())
    else:
        depends_on = store.feature_view(location.name).view.source.depends_on()
    return depends_on


async def levels_for_location(
    location: FeatureLocation,
    store: ContractStore,
    existing_locations: dict[FeatureLocation, int] | None = None,
) -> dict[FeatureLocation, int]:
    depends_on = await location_depends_on(location, store)

    levels = {}
    max_value = 0
    if existing_locations is None:
        existing_locations = {}

    for dep in depends_on:
        if dep in existing_locations:
            set_value = existing_locations[dep]
            levels[dep] = set_value
            max_value = max(max_value, set_value + 1)
        else:
            sub_deps = await levels_for_location(dep, store)
            for key, value in sub_deps.items():
                set_value = max(value, levels.get(key, 0))
                levels[key] = set_value
                max_value = max(max_value, set_value + 1)

    levels[location] = max_value
    return levels


def locations_with_freshness_threshold(
    locations: list[FeatureLocation], store: ContractStore
) -> list[FeatureLocation]:
    locs = []
    for loc in locations:
        if loc.location == "model":
            freshness = store.models[loc.name].predictions_view.acceptable_freshness
        else:
            freshness = store.feature_views[loc.name].acceptable_freshness

        if freshness:
            locs.append(loc)
    return locs


async def depends_on_map(
    locations: list[FeatureLocation], store: ContractStore
) -> dict[FeatureLocation, set[FeatureLocation]]:
    deps = {}
    for location in locations:
        deps[location] = await location_depends_on(location, store)

    return deps


def update_order(levels: dict[FeatureLocation, int]) -> list[list[FeatureLocation]]:
    sorted_levels = list(sorted(levels.items(), key=lambda items: items[1]))

    order = []
    current_stack = []
    current_level = 0

    for key, level in sorted_levels:
        if level == current_level:
            current_stack.append(key)
        else:
            current_level = level
            order.append(current_stack)
            current_stack = [key]

    if current_stack:
        order.append(current_stack)

    return order


@flow
async def update_out_of_date_data(location: str | None = None):
    """
    Updates all data that is out of data based on the `accepted_freshness` threshold.

    If no location is passed in will all views be checked.
    However, you can also pass in a view to only update for a subset.
    E.g: `feature_view:wine` or `model:movie_review_is_negative`
    """
    from src.pipelines.batch_predict import batch_predict_for

    logger = get_run_logger()

    store = await load_store()

    if location:
        loc = FeatureLocation.from_string(location)
        levels = await levels_for_location(loc, store)
    else:
        levels = await levels_for_store(store)

    levels_to_update = locations_with_freshness_threshold(list(levels.keys()), store)

    location_deps = await depends_on_map(levels_to_update, store)
    location_update_order = update_order(
        {loc: val for loc, val in levels.items() if loc in levels_to_update}
    )

    task_map: dict[FeatureLocation, PrefectFuture] = {}

    logger.info(f"Updating info for {levels_to_update}")

    for level in location_update_order:
        for loc in level:
            wait_for = [task_map[dep_loc] for dep_loc in location_deps[loc]]

            if loc.location == "feature_view":
                task_map[loc] = await materialize_view.with_options(
                    name=f"{loc.name}_materialize"
                ).submit(loc.name, store, wait_for=wait_for)
                logger.info(f"Type of task: {type(task_map[loc])} - {task_map[loc]}")
            else:
                task_map[loc] = await batch_predict_for.with_options(
                    name=f"{loc.name}_batch_predict"
                ).submit(loc.name, store, wait_for=wait_for)
                logger.info(f"Type of task: {type(task_map[loc])} - {task_map[loc]}")

    await asyncio.gather(*[task.wait() for task in task_map.values()])
