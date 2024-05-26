from aligned import EventTimestamp, model_contract, UInt64

from src.wine.wine import Wine, dataset_dir
from src.mlflow_model import MLFlowServer

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
    exposed_model=MLFlowServer(
        host="http://wine-model:8080",
        model_name="is_high_quality_wine",
        model_alias="champion",
    ),
    output_source=dataset_dir.csv_at("predictions.csv"),
    dataset_store=dataset_dir.json_at("datasets.json"),
    contacts=["@MatsMoll"],
)
class WineModel:
    wine_id = UInt64().as_entity()

    predicted_at = EventTimestamp()

    predicted_quality = wine.is_high_quality.as_classification_label()
