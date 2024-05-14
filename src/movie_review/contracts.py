from aligned import feature_view, String, Bool, FileSource, model_contract
from aligned.exposed_model.ollama import ollama_embedding_contract
from aligned.exposed_model.mlflow import mlflow_server


dataset_dir = FileSource.directory("data/sentiment")


@feature_view(
    name="movie_review",
    description="Sentiment analysis of text data",
    source=dataset_dir.csv_at("sentiment.csv")
)
class MovieReview:
    file = String().as_entity()

    text = String()

    is_negative = Bool()


review = MovieReview()

MovieReviewEmbedding = ollama_embedding_contract(
    input=review.text,
    entities=review.file,
    model="nomic-embed-text",
    endpoint="http://host.docker.internal:11434",
    contract_name="movie_review_embedding",
)

review_embedding = MovieReviewEmbedding()



@model_contract(
    name="movie_review_is_negative",
    input_features=[
        review_embedding.embedding,
    ],
    output_source=dataset_dir.csv_at("predictions.csv"),
    exposed_model=mlflow_server(
        host="http://movie_review_is_negative:8080",
        model_name="movie_review_is_negative",
        model_alias="champion",
    ),
    dataset_store=dataset_dir.json_at("datasets.json"),
)
class MovieReviewIsNegative:
    file = String().as_entity()
    model_version = String().as_model_version()
    is_negative = review.is_negative.as_classification_label()

