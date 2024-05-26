from pathlib import Path
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

    subprocess.run(
        [
            "mlflow",
            "models",
            "serve",
            "-m",
            uri,
            "--port",
            str(port),
            "--host",
            host,
            "--no-conda",
            "--enable-mlserver",
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
