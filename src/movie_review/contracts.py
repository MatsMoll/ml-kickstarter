from aligned import feature_view, String, Bool, FileSource, model_contract
from aligned.exposed_model.ollama import ollama_embedding_contract
from aligned.exposed_model.mlflow import mlflow_server

dataset_dir = FileSource.directory("data/sentiment")

# @feature_view(
#     name="unannotated_movie_review",
#     source=dataset_dir.csv_at("movie_review.csv")
# )
# class MovieReview:
#     review_id = String().as_entity()
#     text = String()
#     movie_name = String()
#     created_at = EventTimestamp()


@feature_view(
    name="movie_review",
    description="Sentiment analysis of text data",
    source=dataset_dir.csv_at("sentiment.csv"),
    tags=["annotation"],
)
class AnnotatedMovieReview:
    review_id = String().as_entity()
    text = String()
    is_negative = Bool()


# class MovieReviews:
#     review_id = String().as_entity()
#     text = String()
#     created_at = EventTimestamp()
#
#
# """
# Annotert view
#
# Det er et feature view som inneholder annoterte rows, men det er basert på en eksisterende kilde.
#
# Den skal ikke materialize dataen fra kilden, men man skal kunne annotere data fra kilden.
#
# Data to annotate: feature view
# Annotated at: EventTimestamp
# Annotated by: String?
#
# Annotation input: copy av source feature view
#
# Selection:
# - None
# - Range -> Start, end
# - Image section -> Center, height, width -> x, y
#
# Outputen vil være forskjellig basert på oppgaven:
# - Label
# - Segmentation
# """
#
# @annotated_view(
#     view=MovieReviews,
#     annotate_field=MovieReviews().text,
#     source=FileSource.csv_at("")
# )
# class AnnotatedMovieReview:
#     section = ImageSection().is_option() # Or TextSection()
#
#     annotated_at = EventTimestamp()
#     annotated_by = String().as_annotator()
#
#     is_negative = Bool()
#
#     label = String().accepted_values(["A", "b", "c", "d"])
#     segmentations = List(Point())
#

review = AnnotatedMovieReview()

MovieReviewEmbedding = ollama_embedding_contract(
    input=review.text,
    entities=review.review_id,
    model="nomic-embed-text",
    endpoint="http://host.docker.internal:11434",
    contract_name="movie_review_embedding",
    contacts=["@MatsMoll"],
)

review_embedding = MovieReviewEmbedding()


@model_contract(
    name="movie_review_is_negative",
    input_features=[
        review_embedding.embedding,
    ],
    output_source=dataset_dir.csv_at("predictions.csv"),
    exposed_model=mlflow_server(
        host="http://movie-review-is-negative:8080",
        model_name="movie_review_is_negative",
        model_alias="champion",
    ),
    dataset_store=dataset_dir.json_at("datasets.json"),
)
class MovieReviewIsNegative:
    review_id = String().as_entity()
    model_version = String().as_model_version()
    is_negative_pred = review.is_negative.as_classification_label()
