from pathlib import Path
import json
import mlflow
import subprocess
import argparse


def start_mlfow_server(model_name: str, alias: str, port: int, host: str) -> None:
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


def serve_mlflow_model() -> None:
    from watchfiles import run_process

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="The model to serve")
    parser.add_argument("--alias", type=str, default="champion")
    parser.add_argument("--mlflow-dir", type=str, default="mlflow/experiments")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()

    model_name = args.model_name
    alias = args.alias
    mlflow_dir = args.mlflow_dir
    port = args.port
    host = args.host

    model_alias_file = Path(mlflow_dir) / "models" / model_name / "aliases" / alias

    run_process(
        model_alias_file.resolve(),
        target=start_mlfow_server,
        args=(model_name, alias, port, host),
    )


if __name__ == "__main__":
    serve_mlflow_model()
