from datetime import timedelta
from aligned import (
    UInt32,
    feature_view,
    UInt64,
    Int32,
    Float,
    FileSource,
    CsvConfig,
    String,
)
from aligned import EventTimestamp, model_contract
from aligned.exposed_model.mlflow import mlflow_server
import polars as pl

dataset_dir = FileSource.directory("data/wine-quality")

csv_config = CsvConfig(seperator=";")

columns = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]
mapping_keys = {column: column.replace(" ", "_") for column in columns}


def add_hash_column(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(wine_id=pl.concat_str(pl.all()).hash() // 2**32)


@feature_view(
    name="white_wine",
    source=dataset_dir.csv_at(
        "white.csv", csv_config=csv_config, mapping_keys=mapping_keys
    )
    .transform_with_polars(add_hash_column)
    .with_loaded_at(),
    # Materializing as the hash can change based on the selected columns
    materialized_source=dataset_dir.csv_at(
        "white_with_id.csv", csv_config=csv_config, mapping_keys=mapping_keys
    ),
    acceptable_freshness=timedelta(days=30),
)
class WhiteWine:
    """
    Data from the classic [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
    """

    wine_id = UInt32().as_entity()

    loaded_at = EventTimestamp()

    fixed_acidity = Float()
    volatile_acidity = Float()
    citric_acid = Float()
    residual_sugar = Float()
    chlorides = Float()
    free_sulfur_dioxide = Float()
    total_sulfur_dioxide = Float()
    density = Float()
    pH = Float()
    sulphates = Float()
    alcohol = Float()
    quality = Int32()


# Creates a new view with the same schema as WhiteWine
# But swaps out the source with a Red Wine dataset
RedWine = WhiteWine.with_source(  # type: ignore
    "red_wine",
    dataset_dir.csv_at("red.csv", csv_config=csv_config, mapping_keys=mapping_keys)
    .transform_with_polars(add_hash_column)
    .with_loaded_at(),
    # Materializing as the hash can change based on the selected columns
    materialized_source=dataset_dir.csv_at(
        "red_with_id.csv", csv_config=csv_config, mapping_keys=mapping_keys
    ),
)


@feature_view(
    name="wine",
    source=WhiteWine.vstack(RedWine, source_column="origin_view"),  # type: ignore
    acceptable_freshness=timedelta(days=30),
)
class Wine:
    """
    Data from the classic [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
    """

    wine_id = UInt32().as_entity()

    loaded_at = EventTimestamp()

    fixed_acidity = Float()
    volatile_acidity = Float()
    citric_acid = Float()
    residual_sugar = Float()
    chlorides = Float()
    free_sulfur_dioxide = Float()
    total_sulfur_dioxide = Float()
    density = Float()
    pH = Float()
    sulphates = Float()
    alcohol = Float()
    quality = Int32()

    origin_view = String().accepted_values(
        ["feature_view:white_wine", "feature_view:red_wine"]
    )
    is_red_wine = origin_view == "feature_view:red_wine"

    is_high_quality = quality > 6


wine = Wine()


@model_contract(
    name="is_high_quality_wine",
    input_features=[
        wine.fixed_acidity,
        wine.volatile_acidity,
        wine.citric_acid,
        wine.residual_sugar,
        wine.chlorides,
        wine.free_sulfur_dioxide,
        wine.total_sulfur_dioxide,
        wine.density,
        wine.pH,
        wine.sulphates,
        wine.alcohol,
        wine.is_red_wine,
    ],
    exposed_model=mlflow_server(
        host="http://wine-model:8080",
        model_name="is_high_quality_wine",
        model_alias="champion",
    ),
    output_source=dataset_dir.csv_at("predictions.csv"),
    dataset_store=dataset_dir.json_at("datasets.json"),
    contacts=["@MatsMoll"],
    acceptable_freshness=timedelta(days=30),
)
class WineModel:
    wine_id = UInt64().as_entity()
    predicted_at = EventTimestamp()
    predicted_quality = wine.is_high_quality.as_classification_label()
