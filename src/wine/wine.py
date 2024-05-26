from aligned import feature_view, UInt64, Int32, Float, FileSource, CsvConfig, String
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
    return df.with_columns(wine_id=pl.concat_str(pl.all()).hash())


@feature_view(
    name="white_wine",
    source=dataset_dir.csv_at(
        "white.csv", csv_config=csv_config, mapping_keys=mapping_keys
    ).transform_with_polars(add_hash_column),
    # Normalizing as the hash can change based on the selected columns
    materialized_source=dataset_dir.csv_at(
        "white_with_id.csv", csv_config=csv_config, mapping_keys=mapping_keys
    ),
)
class WhiteWine:
    """
    Data from the classic [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
    """

    wine_id = UInt64().as_entity()

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
RedWine = WhiteWine.with_source(
    "red_wine",
    dataset_dir.csv_at(
        "red.csv", csv_config=csv_config, mapping_keys=mapping_keys
    ).transform_with_polars(add_hash_column),
    # Normalizing as the hash can change based on the selected columns
    materialized_source=dataset_dir.csv_at(
        "red_with_id.csv", csv_config=csv_config, mapping_keys=mapping_keys
    ),
)


@feature_view(
    name="wine",
    source=WhiteWine.vstack(RedWine, source_column="origin_view"),
)
class Wine:
    """
    Data from the classic [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
    """

    wine_id = UInt64().as_entity()

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
