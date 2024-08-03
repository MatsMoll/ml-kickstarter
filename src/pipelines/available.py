from pathlib import Path
from prefect import serve
from prefect.deployments.runner import RunnerDeployment

from datetime import timedelta


from src.wine.train import train_wine_model
from src.movie_review.train import train_sentiment
from src.pipelines.materialize import update_out_of_date_data


def all_pipelines() -> list[RunnerDeployment]:
    ml_train_flows = [
        workflow.to_deployment(
            name=workflow.name,
            tags=["ml", "train"],
        )
        for workflow in [train_sentiment, train_wine_model]
    ]
    return ml_train_flows + [
        update_out_of_date_data.to_deployment(
            "update_out_of_date_data", interval=timedelta(minutes=5)
        )
    ]


def listen_to_work() -> None:
    serve(*all_pipelines())  # type: ignore


async def main() -> None:
    from watchfiles import arun_process

    await arun_process(Path("src").resolve(), target=listen_to_work, args=())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
