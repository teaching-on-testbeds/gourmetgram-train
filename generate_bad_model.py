#!/usr/bin/env python3
"""
Generate bad architecture model (ResNet18) for mlops-bad-arch branch.
This script creates a food11.pth with incompatible architecture.
"""

import torch
import torch.nn as nn
from torchvision import models


def main():
    print("=" * 70)
    print("Generating Bad Architecture Model (ResNet18)")
    print("=" * 70)
    
    # Load pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace final layer to output 11 classes
    # ResNet18 has 512 feature dimensions, not 1280 like MobileNetV2
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 11)
    
    print(f"Architecture: ResNet18")
    print(f"Feature dimensions: {num_features} (incompatible with MobileNetV2's 1280)")
    print(f"Output classes: 11")
    
    # Save as food11.pth (will be loaded by flow.py)
    torch.save(model, "food11.pth")
    
    # Check file size
    import os
    size_mb = os.path.getsize("food11.pth") / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
