"""
Image Warp (GPU)
Uses: processing/gpu_utils
Used by: main.py

GPU image warping. RBF interpolation must be on CPU (scipy limitation),
but the actual grid_sample warp is on GPU.
"""

import torch
import numpy as np
from scipy import interpolate
from processing import gpu_utils

# === PARAMETERS ===
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SMOOTHING = 10.0


def warp_image(image_tensor, source_points, destination_points):
    """
    Warp image using displacement field.

    CPU transfer justification:
        - scipy.interpolate.RBFInterpolator has no GPU implementation
        - Points are small (N~64), transfer is negligible
        - Flow field (H*W*2) is computed on CPU then sent to GPU once

    Args:
        image_tensor: [B, C, H, W] float tensor on GPU
        source_points: [N, 2] tensor (y, x) on GPU
        destination_points: [N, 2] tensor (y, x) on GPU

    Returns:
        [B, C, H, W] warped tensor on GPU
    """
    _, _, h, w = image_tensor.shape

    # Transfer points to CPU for RBF (small, ~64 points)
    src_np = source_points.detach().cpu().numpy()
    dst_np = destination_points.detach().cpu().numpy()

    # Skip if NaN
    if np.any(np.isnan(src_np)) or np.any(np.isnan(dst_np)):
        return image_tensor

    displacements = dst_np - src_np

    # RBF interpolation (CPU - no GPU scipy)
    try:
        interpolator = interpolate.RBFInterpolator(
            src_np, displacements, smoothing=SMOOTHING
        )
    except (ValueError, np.linalg.LinAlgError):
        return image_tensor

    # Build dense flow field (CPU)
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    grid = np.stack((grid_y, grid_x), axis=-1).reshape(-1, 2)
    flow = interpolator(grid).reshape((h, w, 2)).astype(np.float32)

    # Build sampling coordinates
    sample_y = grid_y + flow[:, :, 0]
    sample_x = grid_x + flow[:, :, 1]

    # Normalize to [-1, 1] for grid_sample
    sample_x_norm = 2.0 * sample_x / (w - 1) - 1.0
    sample_y_norm = 2.0 * sample_y / (h - 1) - 1.0

    # Transfer grid to GPU (one transfer of H*W*2 floats)
    grid_np = np.stack((sample_x_norm, sample_y_norm), axis=-1).astype(np.float32)
    grid_tensor = torch.from_numpy(grid_np).unsqueeze(0).to(DEVICE)

    # GPU warp
    with torch.no_grad():
        warped = gpu_utils.grid_sample_safe(image_tensor, grid_tensor)

    return warped
