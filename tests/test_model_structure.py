import pytest
import torch


def _load_checkpoint(path: str = "food11.pth"):
    return torch.load(path, weights_only=False, map_location=torch.device("cpu"))


def _coerce_to_model(loaded_obj):
    """Return a torch.nn.Module regardless of whether checkpoint stores a Module or state_dict."""
    if isinstance(loaded_obj, torch.nn.Module):
        return loaded_obj

    # Common patterns: state_dict (OrderedDict) or dict with a nested state_dict
    state_dict = None
    if isinstance(loaded_obj, dict):
        if all(isinstance(k, str) for k in loaded_obj.keys()):
            state_dict = loaded_obj
        elif isinstance(loaded_obj.get("state_dict"), dict):
            state_dict = loaded_obj["state_dict"]

    if state_dict is None:
        pytest.fail(
            "Unsupported checkpoint format; expected torch.nn.Module or state_dict-like dict"
        )

    assert isinstance(state_dict, dict)

    # Build expected architecture (MobileNetV2 classifier for Food-11)
    from torchvision import models
    import torch.nn as nn

    expected_num_classes = 11
    num_classes = expected_num_classes

    if "classifier.1.weight" in state_dict:
        head_w = state_dict["classifier.1.weight"]
        if torch.is_tensor(head_w) and head_w.ndim == 2:
            num_classes = int(head_w.shape[0])

    if num_classes != expected_num_classes:
        pytest.fail(
            f"Checkpoint head has {num_classes} classes; expected {expected_num_classes} for Food-11"
        )

    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes),
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        pytest.fail(f"Unexpected keys when loading checkpoint: {unexpected[:5]}...")
    return model


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
