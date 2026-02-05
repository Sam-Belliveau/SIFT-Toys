"""
Image Warp
Uses: nothing
Used by: main.py

Warps images based on point correspondences.
Interpolates displacement field rather than absolute positions.
"""

import numpy as np
import cv2
from scipy import interpolate

# === PARAMETERS ===
SMOOTHING = 10.0  # RBF smoothing - higher prevents wild extrapolation
DEBUG = True  # Print debug info


def create_displacement_interpolator(source_points, displacements):
    """
    Create interpolator for displacement field.
    Maps source positions -> displacement vectors.
    """
    try:
        return interpolate.RBFInterpolator(
            source_points,
            displacements,
            smoothing=SMOOTHING,
        )
    except (ValueError, np.linalg.LinAlgError) as e:
        if DEBUG:
            print(f"RBF failed: {e}")
        return None


def generate_warp_map(image_shape, source_points, destination_points):
    """
    Generate warp maps for cv2.remap.
    Interpolates displacement field from point correspondences.
    """
    h, w = image_shape[:2]

    # Calculate displacements at control points (in y,x order)
    displacements = destination_points - source_points

    if DEBUG:
        avg_disp = np.mean(np.abs(displacements))
        max_disp = np.max(np.abs(displacements))
        print(
            f"Points: {len(source_points)}, Avg displacement: {avg_disp:.2f}, Max: {max_disp:.2f}"
        )

    # Create displacement interpolator
    interpolator = create_displacement_interpolator(source_points, displacements)

    if interpolator is None:
        # Return identity map on failure
        map_x = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)
        map_y = np.arange(h, dtype=np.float32)[:, np.newaxis].repeat(w, axis=1)
        return map_x, map_y

    # Build coordinate grid (y, x)
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    grid = np.stack((grid_y, grid_x), axis=-1)
    flat_grid = grid.reshape(-1, 2)

    # Interpolate displacement at every pixel
    displacement_field = interpolator(flat_grid).reshape((h, w, 2))

    if DEBUG:
        field_max = np.max(np.abs(displacement_field))
        print(f"Field max displacement: {field_max:.2f}")

    # For cv2.remap: map tells where to sample FROM
    # If we want content to move with the tracked points, we add displacement
    new_y = grid_y + displacement_field[:, :, 0]
    new_x = grid_x + displacement_field[:, :, 1]

    # Clamp to image bounds
    map_y = np.clip(new_y, 0, h - 1).astype(np.float32)
    map_x = np.clip(new_x, 0, w - 1).astype(np.float32)

    return map_x, map_y


def apply_warp(image, map_x, map_y):
    """Apply warp maps to image."""
    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
    )
