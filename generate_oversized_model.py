#!/usr/bin/env python3
"""
Generate oversized model (>200Mi) for mlops-bad-size branch.
This script creates a food11.pth that exceeds Kubernetes memory limits.
"""

import torch
import torch.nn as nn
from torchvision import models


def main():
    print("=" * 70)
    print("Generating Oversized Model (>200Mi)")
    print("=" * 70)
    
    # Load pre-trained MobileNetV2 (compatible architecture)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Replace classifier to output 11 classes
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 11)
    )
    
    # Add massive dummy tensor to inflate model size beyond 200Mi
    target_size_mb = 210
    current_size_mb = 14  # Approximate size of base MobileNetV2
    padding_needed_mb = target_size_mb - current_size_mb
    
    # Calculate number of float32 elements needed (4 bytes per element)
    num_elements = int((padding_needed_mb * 1024 * 1024) / 4)
    
    print(f"Architecture: MobileNetV2 (compatible)")
    print(f"Base model size: ~{current_size_mb} MB")
    print(f"Adding {padding_needed_mb} MB of dummy weights...")
    
    # Add dummy parameter to inflate size
    model.dummy_large_weight = nn.Parameter(
        torch.randn(num_elements), 
        requires_grad=False
    )
    
    # Save as food11.pth (will be loaded by flow.py)
    torch.save(model, "food11.pth")
    
    # Check actual file size
    import os
    size_mb = os.path.getsize("food11.pth") / (1024 * 1024)
    print(f"Final file size: {size_mb:.2f} MB")
    
    if size_mb >= 200:
        print("SUCCESS: Model exceeds 200MB threshold")
    else:
        print("WARNING: Model is smaller than 200MB target")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
