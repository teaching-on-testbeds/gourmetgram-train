import pytest
import torch


def _load_checkpoint(path: str = "food11.pth"):
    return torch.load(path, weights_only=False, map_location=torch.device("cpu"))


def _coerce_to_model(loaded_obj):
    if isinstance(loaded_obj, torch.nn.Module):
        loaded_obj.eval()
        return loaded_obj

    if isinstance(loaded_obj, dict):
        from torchvision import models
        import torch.nn as nn

        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.last_channel
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 11),
        )
        model.load_state_dict(loaded_obj)
        model.eval()
        return model

    pytest.fail(
        f"Unsupported checkpoint type {type(loaded_obj)}; expected state_dict (dict) or torch.nn.Module"
    )


@pytest.fixture(scope="module")
def model():
    # Load model once and share across all tests in this module
    try:
        loaded = _load_checkpoint("food11.pth")
        model = _coerce_to_model(loaded)
        return model.to(torch.device("cpu"))
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")


def test_model_loadable():
    # Test that model file exists and is loadable (doesn't use fixture to verify loading process itself)
    try:
        loaded = _load_checkpoint("food11.pth")
        assert loaded is not None, "Model loaded but is None"
    except FileNotFoundError:
        pytest.fail("Model file 'food11.pth' not found")
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")


@pytest.mark.parametrize("batch_size", [1, 4])
def test_model_input_output_shape(model, batch_size):
    # Expect a Food-11 classifier: 3x224x224 input -> 11 logits
    model.eval()

    x = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)

    assert isinstance(y, torch.Tensor), "Model output is not a torch.Tensor"
    assert y.shape == (batch_size, 11), f"Got output shape {tuple(y.shape)}; expected {(batch_size, 11)}"
