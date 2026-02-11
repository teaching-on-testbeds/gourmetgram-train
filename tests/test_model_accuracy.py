import random


def test_model_accuracy():
    # Simulate probabilistic accuracy: 70% chance of 0.85 (pass), 30% chance of 0.75 (fail)
    if random.random() < 0.7:
        accuracy = 0.85
    else:
        accuracy = 0.75

    print(f"Model accuracy: {accuracy:.2%}")

    assert accuracy >= 0.80, f"Model accuracy {accuracy:.2%} below threshold 80%"
