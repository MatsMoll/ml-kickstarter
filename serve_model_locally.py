from pathlib import Path
import json
import mlflow
import click
import subprocess


@click.group()
def cli() -> None:
    pass


def start_mlfow_server(model_name: str, alias: str, port: int, host: str):
    uri = f"models:/{model_name}@{alias}"
    print(uri)
    try:
        mlflow.models.get_model_info(uri)
    except Exception as e:
        print("Remember to start the tracking server, or train a model first.")
        raise e

    mlserver_settings_dir = Path(f"model-settings/{model_name}")
    mlserver_settings_dir.mkdir(exist_ok=True)

    mlserver_config_path = mlserver_settings_dir / "model-settings.json"
    mlserver_config_path.write_text(
        data=json.dumps(
            {
                "name": model_name,
                "implementation": "src.custom_mlflow_server.MLflowRuntime",
                "parameters": {"uri": uri, "host": host, "http_port": port},
            }
        )
    )

    subprocess.run(
        [
            "mlserver",
            "start",
            mlserver_settings_dir.as_posix(),
        ]
    )


@cli.command()
@click.argument("model_name", type=str)
@click.option("--alias", type=str, default="champion")
@click.option("--mlflow-dir", type=Path, default="mlflow/experiments")
@click.option("--port", type=int, default="8080")
@click.option("--host", type=str, default="0.0.0.0")
def serve_mlflow_model(
    model_name: str, alias: str, mlflow_dir: Path, port: int, host: str
):
    from watchfiles import run_process

    model_alias_file = mlflow_dir / "models" / model_name / "aliases" / alias

    run_process(
        model_alias_file.resolve(),
        target=start_mlfow_server,
        args=(model_name, alias, port, host),
    )


if __name__ == "__main__":
    cli()
