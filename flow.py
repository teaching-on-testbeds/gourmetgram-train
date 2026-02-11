import os
import sys
import time
import subprocess
from pathlib import Path

import mlflow
import torch
from mlflow import MlflowClient
from mlflow.exceptions import RestException


REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATH = REPO_ROOT / "food11.pth"
TESTS_DIR = REPO_ROOT / "tests"
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "GourmetGramFood11Model")


def log(msg: str) -> None:
    print(msg, flush=True)


def emulate_training_and_load_checkpoint() -> object:
    sleep_s = float(os.getenv("EMULATED_TRAINING_SLEEP_SECONDS", "2"))
    log(f"Emulating training (sleep {sleep_s}s)...")
    time.sleep(sleep_s)

    log(f"torch.load({MODEL_PATH.name})...")
    loaded = torch.load(
        str(MODEL_PATH),
        weights_only=False,
        map_location=torch.device("cpu"),
    )

    if not isinstance(loaded, torch.nn.Module):
        raise TypeError(
            f"Expected checkpoint to be a full torch.nn.Module, got {type(loaded)}"
        )
    return loaded


def run_pytest() -> subprocess.CompletedProcess[str]:
    log("Running pytest...")
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS_DIR), "-v", "-s", "--tb=short"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def register_checkpoint_artifact(run_id: str) -> str:
    """Register the logged checkpoint artifact as a new model version."""
    client = MlflowClient()

    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except RestException as e:
        code = getattr(e, "error_code", None)
        if code == "RESOURCE_DOES_NOT_EXIST" or "RESOURCE_DOES_NOT_EXIST" in str(e):
            client.create_registered_model(REGISTERED_MODEL_NAME)
        else:
            raise

    source = f"runs:/{run_id}/model/{MODEL_PATH.name}"
    mv = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=source,
        run_id=run_id,
    )
    return str(mv.version)


def main() -> int:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "food11-classifier"))

    with mlflow.start_run() as run:
        log(f"MLflow run started: {run.info.run_id}")

        emulate_training_and_load_checkpoint()

        log("Logging model artifact to MLflow...")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

        result = run_pytest()
        pytest_log_path = Path("/tmp/pytest_output.txt")
        pytest_log_path.write_text(
            f"Exit Code: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n"
        )
        mlflow.log_artifact(str(pytest_log_path), artifact_path="test_logs")

        if result.returncode == 0:
            log(f"Registering model '{REGISTERED_MODEL_NAME}'...")
            version = register_checkpoint_artifact(run.info.run_id)
            log(f"Registered model version: {version}")
            # Write version for Argo workflow to pick up
            Path("/tmp/model_version").write_text(version)
        else:
            log("=" * 80)
            log("PYTEST FAILED - Test output:")
            log("=" * 80)
            log(result.stdout)
            if result.stderr:
                log("\nSTDERR:")
                log(result.stderr)
            log("=" * 80)

        return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
