"""Model structure validation tests demonstrating pytest fixture usage."""

import pytest
import torch


@pytest.fixture(scope="module")
def model():
    # Load model once and share across all tests in this module
    try:
        model = torch.load("food11.pth", weights_only=False, map_location=torch.device('cpu'))
        return model
    except Exception as e:
        pytest.fail(f"Failed to load model in fixture: {e}")


def test_model_loadable():
    # Test that model file exists and is loadable (doesn't use fixture to verify loading process itself)
    try:
        model = torch.load("food11.pth", weights_only=False, map_location=torch.device('cpu'))
        assert model is not None, "Model loaded but is None"
    except FileNotFoundError:
        pytest.fail("Model file 'food11.pth' not found")
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")


def test_model_parameters(model):
    # Test that model has expected parameter count (2-3M for MobileNetV2 with custom classifier)
    total_params = sum(p.numel() for p in model.parameters())

    min_params = 2_000_000
    max_params = 3_000_000

    assert min_params < total_params < max_params, \
        f"Model has {total_params:,} parameters (expected {min_params:,} to {max_params:,})"

    print(f"✓ Model has {total_params:,} parameters (within expected range)")
