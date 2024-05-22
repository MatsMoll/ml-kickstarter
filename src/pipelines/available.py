from pathlib import Path
from prefect import serve

from src.wine.train import train_wine_model
from src.movie_review.train import train_sentiment

def all_pipelines():
    train_workflows = [
        train_sentiment,
        train_wine_model
    ]

    return [
        workflow.to_deployment(
            name=workflow.name,
            tags=["ml", "train"],
        ) for workflow in train_workflows
    ]


def listen_to_work():
    serve(*all_pipelines()) # type: ignore


async def main():
    from watchfiles import arun_process
    await arun_process(Path().resolve(), target=listen_to_work, args=())

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
