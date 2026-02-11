import re
import time
import subprocess
import torch
import mlflow
from prefect import flow, task, get_run_logger
# NOTE: "from mlflow.tracking import MlflowClient" is a legacy import path.
# In newer MLflow versions (3.x+), the recommended import is:
#   from mlflow import MlflowClient
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

MODEL_PATH = "food11.pth"
MODEL_NAME = "GourmetGramFood11Model"


@task
def load_and_train_model():
    logger = get_run_logger()

    model_path = "food11.pth"
    logger.info(f"Loading model from {model_path}...")
    time.sleep(10)

    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))

    logger.info("Logging model to MLflow...")
    # NOTE: "artifact_path" is deprecated in newer MLflow versions (3.x+).
    # The recommended replacement is the "name" parameter:
    #   mlflow.pytorch.log_model(model, name="model")
    mlflow.pytorch.log_model(model, artifact_path="model")
    return model


@task
def evaluate_model():
    """Run pytest test suite and save complete output as MLFlow artifact."""
    logger = get_run_logger()
    logger.info("Running pytest test suite for model evaluation...")

    try:
        result = subprocess.run(
            ["pytest", "tests/", "-v", "-s", "--tb=short"],
            cwd="/app",
            capture_output=True,
            text=True
        )

        # Save complete pytest output as MLFlow artifact
        full_output = f"Exit Code: {result.returncode}\n"
        full_output += f"Status: {'PASSED' if result.returncode == 0 else 'FAILED'}\n\n"
        full_output += result.stdout
        if result.stderr:
            full_output += f"\n--- STDERR ---\n{result.stderr}"

        pytest_log_path = "/tmp/pytest_output.txt"
        with open(pytest_log_path, "w") as f:
            f.write(full_output)
        mlflow.log_artifact(pytest_log_path, artifact_path="test_logs")

        # Parse test counts from pytest summary line
        tests_passed = 0
        tests_failed = 0
        passed_match = re.search(r'(\d+)\s+passed', result.stdout)
        failed_match = re.search(r'(\d+)\s+failed', result.stdout)
        if passed_match:
            tests_passed = int(passed_match.group(1))
        if failed_match:
            tests_failed = int(failed_match.group(1))

        mlflow.log_metric("tests_passed", tests_passed)
        mlflow.log_metric("tests_failed", tests_failed)
        mlflow.log_metric("tests_total", tests_passed + tests_failed)

        logger.info(f"Test results: {tests_passed} passed, {tests_failed} failed")

        return result.returncode == 0

    except Exception as e:
        logger.error(f"Failed to run pytest: {e}")
        return False


@task
def register_model(model, tests_passed):
    """Register model to MLFlow only if tests passed."""
    logger = get_run_logger()

    if not tests_passed:
        logger.warning("Tests failed - skipping model registration")
        return None

    logger.info("Tests passed - registering model to MLFlow...")

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"

    client = MlflowClient()

    # MLflow requires the Registered Model to exist before creating versions.
    # In a fresh deployment the model may not exist yet, so create it on demand.
    try:
        client.get_registered_model(MODEL_NAME)
    except RestException as e:
        code = getattr(e, "error_code", None)
        if code == "RESOURCE_DOES_NOT_EXIST" or "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.info(f"Registered model '{MODEL_NAME}' not found; creating it...")
            try:
                client.create_registered_model(MODEL_NAME)
            except RestException as create_e:
                create_code = getattr(create_e, "error_code", None)
                # Another run may have created it concurrently.
                if create_code != "RESOURCE_ALREADY_EXISTS" and "RESOURCE_ALREADY_EXISTS" not in str(create_e):
                    raise
        else:
            raise

    mv = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id
    )

    model_version = mv.version
    logger.info(f"Model registered as version {model_version}")

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="development",
        version=model_version
    )
    logger.info(f"Set alias 'development' to version {model_version}")

    return model_version


@flow
def training_flow():
    logger = get_run_logger()

    mlflow.set_experiment("food11-classifier")

    model_version = None
    try:
        with mlflow.start_run() as run:
            logger.info(f"MLFlow run started: {run.info.run_id}")

            model = load_and_train_model()
            tests_passed = evaluate_model()
            model_version = register_model(model, tests_passed)

            if model_version:
                logger.info(f"Pipeline complete. Model version {model_version} ready for build.")
            else:
                logger.info("Pipeline complete. No model registered (tests failed).")
    except Exception as e:
        # Ensure Argo always finds the output parameter file; an empty value
        # prevents downstream build steps while still surfacing logs.
        logger.error(f"Training flow failed; no model will be built. Error: {e}")
        model_version = None
    finally:
        # Write model version to file for Argo workflow to read
        with open("/tmp/model_version", "w") as f:
            f.write(str(model_version) if model_version else "")


if __name__ == "__main__":
    training_flow()
