"""
SIFT Feature Detector (GPU)
Uses: profiler
Used by: main.py

Kornia SIFT with MPS compatibility via grid_sample patch.
"""

import torch
import torch.nn.functional as F
import kornia.feature as KF
from profiler import profiler

# === PARAMETERS ===
DEFAULT_MAX_FEATURES = 64
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Patch grid_sample to use 'zeros' instead of 'border' for MPS compatibility
original_grid_sample = F.grid_sample


def patched_grid_sample(
    input,
    grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=None,
):
    if padding_mode == "border":
        padding_mode = "zeros"
    return original_grid_sample(
        input,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


F.grid_sample = patched_grid_sample


class SIFTDetector:
    """GPU-accelerated SIFT detector."""

    def __init__(self):
        self.detector = KF.SIFTFeature(num_features=DEFAULT_MAX_FEATURES).to(DEVICE)
        self.max_features = DEFAULT_MAX_FEATURES

    def detect(
        self,
        gray_tensor,
        max_features=DEFAULT_MAX_FEATURES,
    ):
        """Extract SIFT keypoints and descriptors."""
        with profiler.section("rebuild_detector"):
            if max_features != self.max_features:
                self.detector = KF.SIFTFeature(num_features=max_features).to(DEVICE)
                self.max_features = max_features

        with profiler.section("kornia_detect"):
            with torch.no_grad():
                lafs, responses, descriptors = self.detector(gray_tensor)

        with profiler.section("extract_keypoints"):
            if lafs.shape[1] == 0:
                return torch.zeros((0, 2), device=DEVICE), None

            keypoints_xy = lafs[0, :, :, 2]
            keypoints = torch.flip(keypoints_xy, dims=[1])

        return keypoints, descriptors[0]
