"""
Image Warp (GPU)
Uses: processing/gpu_utils, profiler
Used by: main.py

GPU image warping. RBF interpolation must be on CPU (scipy limitation),
but the actual grid_sample warp is on GPU.
"""

import torch
import numpy as np
from scipy import interpolate
from processing import gpu_utils
from profiler import profiler

# === PARAMETERS ===
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SMOOTHING = 10.0


def warp_image(
    image_tensor,
    source_points,
    destination_points,
):
    """
    Warp image using displacement field.

    CPU transfer justification:
        - scipy.interpolate.RBFInterpolator has no GPU implementation
        - Points are small (N~64), transfer is negligible
        - Flow field (H*W*2) computed on CPU then sent to GPU once
    """
    _, _, h, w = image_tensor.shape

    with profiler.section("to_cpu"):
        src_np = source_points.detach().cpu().numpy()
        dst_np = destination_points.detach().cpu().numpy()

    if np.any(np.isnan(src_np)) or np.any(np.isnan(dst_np)):
        return image_tensor

    with profiler.section("displacement"):
        displacements = dst_np - src_np

    with profiler.section("rbf_create"):
        try:
            interpolator = interpolate.RBFInterpolator(
                src_np,
                displacements,
                smoothing=SMOOTHING,
            )
        except (ValueError, np.linalg.LinAlgError):
            return image_tensor

    with profiler.section("build_grid"):
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        grid = np.stack((grid_y, grid_x), axis=-1).reshape(-1, 2)

    with profiler.section("rbf_eval"):
        flow = interpolator(grid).reshape((h, w, 2)).astype(np.float32)

    with profiler.section("sample_coords"):
        sample_y = grid_y + flow[:, :, 0]
        sample_x = grid_x + flow[:, :, 1]
        sample_x_norm = 2.0 * sample_x / (w - 1) - 1.0
        sample_y_norm = 2.0 * sample_y / (h - 1) - 1.0
        grid_np = np.stack((sample_x_norm, sample_y_norm), axis=-1).astype(np.float32)

    with profiler.section("to_gpu"):
        grid_tensor = torch.from_numpy(grid_np).unsqueeze(0).to(DEVICE)

    with profiler.section("grid_sample"):
        with torch.no_grad():
            warped = gpu_utils.grid_sample_safe(image_tensor, grid_tensor)

    return warped
