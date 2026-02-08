"""
Image Warp (CPU)
Uses: profiler
Used by: main.py

RBF displacement field evaluated on coarse grid, upsampled with bicubic.
"""

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from profiler import profiler
from params import params

FLOW_DOWNSAMPLE = 8


def warp_image(
    image,
    source_points,
    destination_points,
):
    h, w = image.shape[:2]
    empty_flow = np.zeros((h, w, 2), dtype=np.float32)

    with profiler.section("nan_check"):
        if np.any(np.isnan(source_points)) or np.any(np.isnan(destination_points)):
            return image, empty_flow

    with profiler.section("displacement"):
        displacements = source_points - destination_points

    with profiler.section("interpolator"):
        try:
            corners = np.array(
                [[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]], dtype=np.float32
            )
            corner_disp = np.zeros((4, 2), dtype=np.float32)

            all_points = np.vstack([destination_points, corners])
            all_disps = np.vstack([displacements, corner_disp])

            _, unique_indices = np.unique(
                (all_points / 10).astype(np.int32),
                axis=0,
                return_index=True,
            )
            all_points = all_points[unique_indices]
            all_disps = all_disps[unique_indices]

            if len(all_points) < 4:
                return image, empty_flow

            interpolator = RBFInterpolator(
                all_points,
                all_disps,
                kernel="multiquadric",
                epsilon=500,
                smoothing=params["rbf_smoothing"],
            )
        except Exception as e:
            print(f"Interpolator failed: {e}")
            return image, empty_flow

    with profiler.section("build_grid"):
        coarse_y, coarse_x = np.mgrid[0:h:FLOW_DOWNSAMPLE, 0:w:FLOW_DOWNSAMPLE]
        small_h, small_w = coarse_y.shape
        coarse_grid = (
            np.stack((coarse_y, coarse_x), axis=-1).reshape(-1, 2).astype(np.float32)
        )

    with profiler.section("interp_eval"):
        coarse_flow = (
            interpolator(coarse_grid).reshape((small_h, small_w, 2)).astype(np.float32)
        )

    with profiler.section("upsample_flow"):
        flow_y = cv2.resize(
            coarse_flow[:, :, 0],
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        flow_x = cv2.resize(
            coarse_flow[:, :, 1],
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )

    with profiler.section("bilateral"):
        d = params["bilateral_sigma"]
        if d > 0:
            d = d | 1  # ensure odd
            sigma = d / 2.0
            guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            flow_y = cv2.ximgproc.jointBilateralFilter(
                guide,
                flow_y,
                d=d,
                sigmaColor=sigma,
                sigmaSpace=sigma,
            )
            flow_x = cv2.ximgproc.jointBilateralFilter(
                guide,
                flow_x,
                d=d,
                sigmaColor=sigma,
                sigmaSpace=sigma,
            )

    with profiler.section("sample_coords"):
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        sample_y = (grid_y + flow_y).astype(np.float32)
        sample_x = (grid_x + flow_x).astype(np.float32)

    with profiler.section("remap"):
        warped = cv2.remap(
            image,
            sample_x,
            sample_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    flow = np.stack([flow_y, flow_x], axis=-1)
    return warped, flow
