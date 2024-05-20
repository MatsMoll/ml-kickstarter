from pathlib import Path
from prefect import serve

from src.wine.train import train_wine_model
from src.movie_review.train import train_sentiment, train_sentiment_test

def all_pipelines():
    return [
        train_sentiment.to_deployment(train_sentiment.name, tags=["ml", "movie_review_is_negative"]),
        train_sentiment_test.to_deployment(train_sentiment_test.name, tags=["ml", "movie_review_is_negative"]),
        train_wine_model.to_deployment(train_wine_model.name, tags=["ml", "is_high_quality_wine"]),
    ]


def listen_to_work():
    serve(*all_pipelines()) # type: ignore


async def main():
    from watchfiles import arun_process
    await arun_process(Path().resolve(), target=listen_to_work, args=())

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
