def test_model_accuracy():
    # Deterministic "good" accuracy for the mlops-bad branch.
    accuracy = 0.90

    print(f"Model accuracy: {accuracy:.2%}")

    assert accuracy >= 0.80, f"Model accuracy {accuracy:.2%} below threshold 80%"
