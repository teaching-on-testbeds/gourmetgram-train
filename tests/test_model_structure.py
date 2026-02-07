"""Model structure validation tests demonstrating pytest fixture usage."""

import pytest
import torch
import torch.nn as nn


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


def test_model_architecture(model):
    # Test that model is MobileNetV2-based (catches ResNet or VGG deployments)
    assert hasattr(model, 'features'), \
        "Model missing 'features' attribute (expected for MobileNetV2)"

    assert isinstance(model.features, nn.Sequential), \
        f"Model features is {type(model.features)}, expected nn.Sequential"

    assert hasattr(model, 'classifier'), \
        "Model missing 'classifier' attribute (expected for MobileNetV2)"

    print(f"✓ Model has correct MobileNetV2 structure")


def test_model_parameters(model):
    # Test that model has expected parameter count (2-3M for MobileNetV2 with custom classifier)
    total_params = sum(p.numel() for p in model.parameters())

    min_params = 2_000_000
    max_params = 3_000_000

    assert min_params < total_params < max_params, \
        f"Model has {total_params:,} parameters (expected {min_params:,} to {max_params:,})"

    print(f"✓ Model has {total_params:,} parameters (within expected range)")


def test_model_output_shape(model):
    # Test that model outputs correct shape for 11 food classes (224x224 RGB input)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (1, 11)
    assert output.shape == expected_shape, \
        f"Model output shape {output.shape} != expected {expected_shape}"

    print(f"✓ Model output shape is correct: {output.shape}")


def test_model_inference_runs(model):
    # Test that model can run inference without crashing and produces reasonable outputs
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    assert torch.isfinite(output).all(), \
        "Model output contains NaN or Inf values"

    assert not torch.allclose(output, torch.zeros_like(output)), \
        "Model output is all zeros (dead model)"

    prediction = output.argmax(dim=1).item()
    assert 0 <= prediction < 11, \
        f"Model prediction {prediction} out of range [0, 11)"

    print(f"✓ Model inference runs successfully, predicted class: {prediction}")
