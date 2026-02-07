import sys
import time
import subprocess
import torch
import mlflow
from prefect import flow, task, get_run_logger
from mlflow.tracking import MlflowClient

MODEL_PATH = "food11.pth"
MODEL_NAME = "GourmetGramFood11Model"


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
    # Run pytest test suite and capture complete output as MLFlow artifact
    logger = get_run_logger()
    logger.info("Running pytest test suite for model evaluation...")

    try:
        # Execute pytest and capture complete output
        result = subprocess.run(
            ["pytest", "tests/", "-v", "--tb=short"],
            cwd="/app",
            capture_output=True,
            text=True
        )

        # Create comprehensive pytest output artifact
        full_output = "=" * 80 + "\n"
        full_output += "PYTEST TEST EXECUTION LOG\n"
        full_output += "=" * 80 + "\n\n"
        full_output += f"Exit Code: {result.returncode}\n"
        full_output += f"Status: {'PASSED' if result.returncode == 0 else 'FAILED'}\n\n"

        full_output += "=" * 80 + "\n"
        full_output += "STDOUT\n"
        full_output += "=" * 80 + "\n"
        full_output += result.stdout + "\n\n"

        if result.stderr:
            full_output += "=" * 80 + "\n"
            full_output += "STDERR\n"
            full_output += "=" * 80 + "\n"
            full_output += result.stderr + "\n\n"

        # Save pytest output as artifact
        pytest_log_path = "/tmp/pytest_output.txt"
        with open(pytest_log_path, "w") as f:
            f.write(full_output)

        mlflow.log_artifact(pytest_log_path, artifact_path="test_logs")
        logger.info("Pytest output saved as artifact: test_logs/pytest_output.txt")

        # Also log to Prefect logger so it appears in workflow logs
        logger.info(f"\n{'='*60}\nPytest Output:\n{'='*60}\n{result.stdout}")

        # Parse test results from pytest output
        tests_passed = 0
        tests_failed = 0
        tests_total = 0

        for line in result.stdout.split('\n'):
            if 'passed' in line.lower() or 'failed' in line.lower():
                # Parse line like "5 passed in 2.34s" or "1 failed, 4 passed in 2.34s"
                import re
                passed_match = re.search(r'(\d+)\s+passed', line)
                failed_match = re.search(r'(\d+)\s+failed', line)

                if passed_match:
                    tests_passed = int(passed_match.group(1))
                if failed_match:
                    tests_failed = int(failed_match.group(1))

        tests_total = tests_passed + tests_failed

        # Log metrics to MLFlow
        mlflow.log_metric("tests_passed", tests_passed)
        mlflow.log_metric("tests_failed", tests_failed)
        mlflow.log_metric("tests_total", tests_total)
        mlflow.log_param("pytest_status", "passed" if result.returncode == 0 else "failed")

        logger.info(f"Test Results: {tests_passed}/{tests_total} passed, {tests_failed}/{tests_total} failed")

        return result.returncode == 0

    except Exception as e:
        logger.error(f"Failed to run pytest: {str(e)}")

        # Log error as artifact too
        error_log = f"ERROR: Pytest execution failed\n\n{str(e)}\n"
        error_log_path = "/tmp/pytest_error.txt"
        with open(error_log_path, "w") as f:
            f.write(error_log)
        mlflow.log_artifact(error_log_path, artifact_path="test_logs")

        mlflow.log_param("pytest_status", "error")
        return False


@task
def register_model(model, tests_passed):
    # Register model to MLFlow only if tests passed
    logger = get_run_logger()

    if not tests_passed:
        logger.warning("Tests failed - skipping model registration")
        mlflow.log_param("model_registered", False)
        return None

    logger.info("Tests passed - registering model to MLFlow...")

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"

    client = MlflowClient()
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

    mlflow.log_param("model_registered", True)
    mlflow.log_param("model_version", model_version)

    return model_version


@flow
def training_flow():
    # Main training flow with comprehensive logging
    logger = get_run_logger()

    mlflow.set_experiment("food11-classifier")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLFlow run started: {run_id}")

        import datetime
        start_time = datetime.datetime.now(datetime.timezone.utc)
        mlflow.log_param("flow_start_time", start_time.isoformat())

        model = load_and_train_model()
        tests_passed = evaluate_model()
        model_version = register_model(model, tests_passed)

        end_time = datetime.datetime.now(datetime.timezone.utc)
        duration = (end_time - start_time).total_seconds()
        mlflow.log_param("flow_end_time", end_time.isoformat())
        mlflow.log_metric("flow_duration_seconds", duration)

        # Create flow execution summary artifact
        summary = "=" * 80 + "\n"
        summary += "TRAINING FLOW EXECUTION SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        summary += f"Run ID: {run_id}\n"
        summary += f"Start Time: {start_time.isoformat()}\n"
        summary += f"End Time: {end_time.isoformat()}\n"
        summary += f"Duration: {duration:.2f} seconds\n\n"
        summary += f"Tests Passed: {'Yes' if tests_passed else 'No'}\n"
        summary += f"Model Registered: {'Yes' if model_version else 'No'}\n"

        if model_version:
            summary += f"Model Version: {model_version}\n"
            summary += f"Model Alias: development\n"

        summary += "\n" + "=" * 80 + "\n"
        summary += "ARTIFACTS\n"
        summary += "=" * 80 + "\n"
        summary += "- test_logs/pytest_output.txt: Complete pytest execution log\n"
        summary += "- model/: Trained model artifacts\n"
        summary += "- logs/flow_summary.txt: This summary file\n"

        summary += "\n" + "=" * 80 + "\n"
        summary += "NEXT STEPS\n"
        summary += "=" * 80 + "\n"

        if model_version:
            summary += f"1. Review test results in test_logs/pytest_output.txt\n"
            summary += f"2. Model version {model_version} is ready for container build\n"
            summary += f"3. Workflow will trigger container build automatically\n"
        else:
            summary += "1. Review test failures in test_logs/pytest_output.txt\n"
            summary += "2. Fix model issues and retrain\n"
            summary += "3. Model was NOT registered due to test failures\n"

        summary_path = "/tmp/flow_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        mlflow.log_artifact(summary_path, artifact_path="logs")

        logger.info("Flow summary saved as artifact: logs/flow_summary.txt")
        logger.info(f"\n{summary}")

        # Write model version to file for Argo workflow to read
        if model_version:
            with open("/tmp/model_version", "w") as f:
                f.write(str(model_version))
            logger.info(f"Model version written to /tmp/model_version: {model_version}")
        else:
            with open("/tmp/model_version", "w") as f:
                f.write("")
            logger.info("No model version (tests failed)")


if __name__ == "__main__":
    training_flow()
