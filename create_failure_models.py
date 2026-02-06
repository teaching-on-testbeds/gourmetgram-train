#!/usr/bin/env python3
"""
Create failure scenario models for testing MLOps pipeline robustness.

This script generates two test model artifacts:
1. bad_model.pth - ResNet18 architecture (incompatible with MobileNetV2 app)
2. oversized_model.pth - Artificially inflated model >200Mi (triggers OOM)

Usage:
    python create_failure_models.py

Requirements:
    torch, torchvision
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os
from oversized_model import OversizedModel

# Target size for oversized model (in bytes)
TARGET_SIZE_BYTES = 210 * 1024 * 1024  # 210Mi (exceeds 200Mi threshold)


def create_bad_model():
    """
    Create a ResNet18 model (incompatible architecture).

    The app expects MobileNetV2 with a custom classifier (1280 → 11).
    This ResNet18 has a different feature extractor dimension (512 → 11),
    which will cause shape mismatch errors when the app tries to load it.

    Returns:
        nn.Module: ResNet18 model with custom classifier for 11 food classes
    """
    print("Creating bad_model.pth (ResNet18 architecture)...")

    # Load pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace final layer to output 11 classes (food categories)
    # ResNet18 has 512 feature dimensions, not 1280 like MobileNetV2
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 11)

    print(f"  Architecture: ResNet18")
    print(f"  Feature dimensions: {num_features} (incompatible with MobileNetV2's 1280)")
    print(f"  Output classes: 11")

    # Save the model
    torch.save(model, "bad_model.pth")

    # Check file size
    size_mb = os.path.getsize("bad_model.pth") / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")
    print()

    return model


def create_oversized_model():
    """
    Create an artificially inflated model >200Mi.

    Starts with the valid food11.pth model, then appends large dummy tensors
    to inflate the file size beyond Kubernetes memory limits (256Mi pods).
    This simulates what happens when:
    - Model weights become bloated during training
    - Large embedding layers are added
    - Model pruning/quantization is skipped

    Returns:
        nn.Module: Original model with dummy weight padding
    """
    print("Creating oversized_model.pth (>200Mi)...")

    # Load the original valid model
    if not os.path.exists("food11.pth"):
        raise FileNotFoundError("food11.pth not found. Cannot create oversized model.")

    original_model = torch.load("food11.pth", weights_only=False, map_location=torch.device('cpu'))
    original_size_mb = os.path.getsize("food11.pth") / (1024 * 1024)
    print(f"  Base model size: {original_size_mb:.2f} MB")

    # Calculate how much padding we need
    current_size = os.path.getsize("food11.pth")
    padding_needed = TARGET_SIZE_BYTES - current_size

    # Create a large dummy tensor as padding
    # Using float32 (4 bytes per element)
    num_elements = padding_needed // 4
    padding_tensor = torch.randn(num_elements)

    print(f"  Adding {padding_needed / (1024 * 1024):.2f} MB of dummy weights...")

    # Wrap the original model with padding
    oversized = OversizedModel(original_model, padding_tensor)

    # Save the oversized model
    torch.save(oversized, "oversized_model.pth")

    # Verify file size
    final_size_mb = os.path.getsize("oversized_model.pth") / (1024 * 1024)
    print(f"  Final file size: {final_size_mb:.2f} MB")

    if final_size_mb < 200:
        print(f"  WARNING: Model is smaller than 200MB target!")
    else:
        print(f"  SUCCESS: Model exceeds 200MB threshold")

    print()

    return oversized


def verify_models():
    """
    Verify that both models can be loaded with PyTorch.

    This doesn't test the failure scenarios themselves (those are tested
    in the workflow), just confirms the .pth files are valid PyTorch models.
    """
    print("Verifying generated models...")

    # Test bad_model.pth
    try:
        bad_model = torch.load("bad_model.pth", weights_only=False, map_location=torch.device('cpu'))
        print("  ✓ bad_model.pth loads successfully")
        print(f"    Type: {type(bad_model).__name__}")
    except Exception as e:
        print(f"  ✗ bad_model.pth failed to load: {e}")
        return False

    # Test oversized_model.pth
    try:
        oversized_model = torch.load("oversized_model.pth", weights_only=False, map_location=torch.device('cpu'))
        print("  ✓ oversized_model.pth loads successfully")
        print(f"    Type: {type(oversized_model).__name__}")
    except Exception as e:
        print(f"  ✗ oversized_model.pth failed to load: {e}")
        return False

    print()
    return True


def main():
    """Generate both failure scenario models and verify them."""
    print("=" * 70)
    print("Generating Failure Scenario Models for MLOps Tutorial")
    print("=" * 70)
    print()

    # Create both models
    create_bad_model()
    create_oversized_model()

    # Verify they load correctly
    if verify_models():
        print("=" * 70)
        print("SUCCESS: All failure models generated and verified")
        print("=" * 70)
        print()
        print("Files created:")
        print("  - bad_model.pth: ResNet18 (incompatible with MobileNetV2 app)")
        print("  - oversized_model.pth: Inflated model >200Mi (triggers OOM)")
        print()
        print("These models will be used by flow.py based on the 'scenario' parameter.")
        return 0
    else:
        print("ERROR: Model verification failed")
        return 1


if __name__ == "__main__":
    exit(main())
