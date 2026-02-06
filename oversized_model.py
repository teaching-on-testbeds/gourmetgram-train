"""
Module defining OversizedModel class.

This must be a separate module so the class can be unpickled when loading
oversized_model.pth. When torch.save() serializes the model, it stores
the module path, so we need the class definition available at that path.
"""

import torch.nn as nn


class OversizedModel(nn.Module):
    """
    Wrapper that adds dummy padding to a base model to inflate file size.

    This is used to simulate oversized models that exceed Kubernetes memory limits.
    The padding is stored as a buffer and ignored during forward pass.
    """
    def __init__(self, base_model, padding):
        super().__init__()
        self.base_model = base_model
        # Register padding as buffer so it's saved with the model
        self.register_buffer('_dummy_padding', padding)

    def forward(self, x):
        # Forward pass ignores padding, uses base model
        return self.base_model(x)
