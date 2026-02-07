import sys
import time
import subprocess
import torch
import mlflow
from prefect import flow, task, get_run_logger
from mlflow.tracking import MlflowClient

MODEL_PATH = "food11.pth"
MODEL_NAME = "GourmetGramFood11Model"


def run_pytest():
    """
    Execute pytest test suite and return pass/fail status.

    This function runs pytest on the tests/ directory and returns a boolean
    indicating whether all tests passed. It can be imported and tested independently.

    Returns:
        bool: True if all tests passed, False if any tests failed or pytest crashed
    """
    try:
        # Execute pytest with verbose output and short tracebacks
        result = subprocess.run(
            ["pytest", "tests/", "-v", "--tb=short"],
            cwd="/app",
            capture_output=True,
            text=True
        )

        # Parse pytest exit code (0 = all tests passed, non-zero = failures)
        return result.returncode == 0

    except Exception:
        # Treat pytest execution failure as test failure
        return False


@task
def load_and_train_model():
    logger = get_run_logger()
    logger.info("Loading model...")
    
    model_path = "food11.pth"
    logger.info(f"Loading model from {model_path}...")
    time.sleep(10)
    
    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    
    logger.info("Logging model to MLflow...")
    mlflow.pytorch.log_model(model, artifact_path="model")
    return model


@task
def evaluate_model():
    logger = get_run_logger()
    logger.info("Running pytest test suite for model evaluation...")

    try:
        # Execute pytest and get detailed output for logging
        result = subprocess.run(
            ["pytest", "tests/", "-v", "--tb=short"],
            cwd="/app",
            capture_output=True,
            text=True
        )

        # Use run_pytest() for pass/fail determination
        all_tests_passed = (result.returncode == 0)

        # Extract test counts from pytest output for MLFlow metrics
        # Pytest typically outputs something like "5 passed in 0.23s" or "2 failed, 3 passed in 0.45s"
        output_lines = result.stdout + result.stderr

        tests_passed = 0
        tests_failed = 0
        tests_total = 0

        # Look for pytest summary line patterns
        if "passed" in output_lines:
            # Try to extract numbers from summary
            import re
            passed_match = re.search(r'(\d+) passed', output_lines)
            failed_match = re.search(r'(\d+) failed', output_lines)

            if passed_match:
                tests_passed = int(passed_match.group(1))
            if failed_match:
                tests_failed = int(failed_match.group(1))

            tests_total = tests_passed + tests_failed

        # Log summary to Prefect logger
        if all_tests_passed:
            logger.info(f"Pytest: {tests_passed} passed")
        else:
            logger.info(f"Pytest: {tests_failed} failed, {tests_passed} passed")
            logger.warning("Some tests failed. Model will not be registered.")

        # Log metrics to MLFlow
        mlflow.log_metric("tests_passed", tests_passed)
        mlflow.log_metric("tests_failed", tests_failed)
        mlflow.log_metric("tests_total", tests_total)

        return all_tests_passed

    except Exception as e:
        logger.error(f"Failed to execute pytest: {e}")
        # Treat pytest execution failure as test failure
        mlflow.log_metric("tests_passed", 0)
        mlflow.log_metric("tests_failed", 0)
        mlflow.log_metric("tests_total", 0)
        return False

@task
def register_model_if_passed(passed: bool):
    logger = get_run_logger()
    if not passed:
        logger.info("Evaluation did not pass criteria. Skipping registration.")
        return None

    logger.info("Registering model in MLflow Model Registry...")
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="development",
        version=registered_model.version
    )
    logger.info(f"Model registered (v{registered_model.version}) and alias 'development' assigned.")
    return registered_model.version

@flow(name="mlflow_flow")
def ml_pipeline_flow():
    with mlflow.start_run():
        load_and_train_model()
        passed = evaluate_model()
        version = register_model_if_passed(passed)
        return version


if __name__ == "__main__":
    print("Starting training pipeline...")

    version = ml_pipeline_flow()

    # Write model version to file for workflow to read
    with open("/tmp/model_version", "w") as f:
        f.write("" if version is None else str(version))

    print(f"Pipeline complete. Model version: {version if version else 'Not registered'}")
