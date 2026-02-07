"""
Image Warp (CPU)
Uses: profiler
Used by: main.py

Linear interpolation-based image warping using scipy Delaunay triangulation.
"""

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from profiler import profiler


def warp_image(
    image,
    source_points,
    destination_points,
):
    """
    Warp image using linear interpolation displacement field.

    Args:
        image: HxWxC uint8 numpy array
        source_points: Nx2 array (y, x) - detected positions
        destination_points: Nx2 array (y, x) - tracked positions
    """
    h, w = image.shape[:2]

    with profiler.section("nan_check"):
        if np.any(np.isnan(source_points)) or np.any(np.isnan(destination_points)):
            return image

    with profiler.section("displacement"):
        displacements = source_points - destination_points

    with profiler.section("interpolator"):
        try:
            # Add corners to ensure full coverage
            corners = np.array(
                [[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]], dtype=np.float32
            )
            corner_disp = np.zeros((4, 2), dtype=np.float32)

            all_points = np.vstack([destination_points, corners])
            all_disps = np.vstack([displacements, corner_disp])

            # Remove duplicate points (within 10 pixels) to avoid singular matrix
            _, unique_indices = np.unique(
                (all_points / 10).astype(np.int32),
                axis=0,
                return_index=True,
            )
            all_points = all_points[unique_indices]
            all_disps = all_disps[unique_indices]

            if len(all_points) < 4:
                return image

            interpolator = RBFInterpolator(
                all_points,
                all_disps,
                kernel="multiquadric",
                epsilon=1.0,
            )
        except Exception as e:
            print(f"Interpolator failed: {e}")
            return image

    with profiler.section("build_grid"):
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        grid = np.stack((grid_y, grid_x), axis=-1).reshape(-1, 2).astype(np.float32)

    with profiler.section("interp_eval"):
        flow = interpolator(grid).reshape((h, w, 2)).astype(np.float32)

    with profiler.section("sample_coords"):
        sample_y = (grid_y + flow[:, :, 0]).astype(np.float32)
        sample_x = (grid_x + flow[:, :, 1]).astype(np.float32)

    with profiler.section("remap"):
        warped = cv2.remap(
            image,
            sample_x,
            sample_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return warped
