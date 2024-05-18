from pathlib import Path
from prefect import serve

from src.pipelines.train import train_sentiment, train_sentiment_test

def all_pipelines():
    return [
        train_sentiment.to_deployment("train_sentiment", tags=["ml"]),
        train_sentiment_test.to_deployment("train_sentiment_test", tags=["ml"]),
    ]


def listen_to_work():
    serve(*all_pipelines()) # type: ignore


async def main():
    from watchfiles import arun_process
    await arun_process(Path().resolve(), target=listen_to_work, args=())

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
