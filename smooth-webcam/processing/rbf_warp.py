"""
RBF Warp
Uses: nothing
Used by: main.py

Radial Basis Function interpolation for smooth image warping.
"""

import numpy as np
import cv2
from scipy import interpolate

# === PARAMETERS ===
RBF_KERNEL = "thin_plate_spline"  # Options: multiquadric, thin_plate_spline, etc.


def _create_interpolator(source_points, destination_points):
    """Create RBF interpolator from point correspondences."""
    try:
        return interpolate.RBFInterpolator(
            source_points,
            destination_points,
        )
    except ValueError as e:
        print(
            f"Failed to create interpolator: "
            f"({source_points.shape} -> {destination_points.shape})"
        )
        print(e)
        return lambda x: x


def generate_warp_map(image_shape, source_points, destination_points):
    """
    Generate warp maps for cv2.remap.

    Returns (map_x, map_y) flow fields.
    """
    h, w = image_shape[:2]

    grid_y, grid_x = np.mgrid[0:h, 0:w]

    interpolator = _create_interpolator(source_points, destination_points)

    grid = np.stack((grid_y, grid_x), axis=-1)
    flow = interpolator(grid.reshape(-1, 2)).reshape((h, w, 2))

    map_x = flow[:, :, 1].astype(np.float32)
    map_y = flow[:, :, 0].astype(np.float32)

    return map_x, map_y


def apply_warp(image, map_x, map_y):
    """Apply warp maps to image using cv2.remap."""
    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
    )
