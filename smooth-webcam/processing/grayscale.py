"""
Grayscale Conversion (GPU)
Uses: nothing
Used by: main.py
"""

import torch
import kornia.color as KC

# === PARAMETERS ===
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def convert(image_tensor):
    """
    Convert RGB tensor to grayscale.

    Args:
        image_tensor: [B, 3, H, W] float tensor

    Returns:
        [B, 1, H, W] float tensor (grayscale)
    """
    return KC.rgb_to_grayscale(image_tensor)
