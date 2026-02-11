import pytest
import torch


def _load_checkpoint(path: str = "food11.pth"):
    return torch.load(path, weights_only=False, map_location=torch.device("cpu"))


@pytest.fixture(scope="module")
def model():
    # Load model once and share across all tests in this module
    try:
        loaded = _load_checkpoint("food11.pth")
        if not isinstance(loaded, torch.nn.Module):
            pytest.fail(
                f"Checkpoint must be a full torch.nn.Module; got {type(loaded)}"
            )
        loaded.eval()
        return loaded.to(torch.device("cpu"))
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
