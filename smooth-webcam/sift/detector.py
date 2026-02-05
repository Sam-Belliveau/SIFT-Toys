"""
SIFT Feature Detector (GPU)
Uses: nothing
Used by: main.py

Kornia SIFT with MPS compatibility via grid_sample patch.
"""

import torch
import torch.nn.functional as F
import kornia.feature as KF

# === PARAMETERS ===
DEFAULT_MAX_FEATURES = 64
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Patch grid_sample to use 'zeros' instead of 'border' for MPS compatibility
_original_grid_sample = F.grid_sample


def _patched_grid_sample(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=None
):
    if padding_mode == "border":
        padding_mode = "zeros"  # MPS doesn't support border, zeros works fine
    return _original_grid_sample(
        input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )


F.grid_sample = _patched_grid_sample


class SIFTDetector:
    """GPU-accelerated SIFT detector."""

    def __init__(self):
        self.detector = KF.SIFTFeature(num_features=DEFAULT_MAX_FEATURES).to(DEVICE)
        self.max_features = DEFAULT_MAX_FEATURES

    def detect(self, gray_tensor, max_features=DEFAULT_MAX_FEATURES):
        """
        Extract SIFT keypoints and descriptors.

        Args:
            gray_tensor: [B, 1, H, W] float tensor on GPU

        Returns:
            (keypoints, descriptors): tensors on GPU
                - keypoints: [N, 2] as (y, x)
                - descriptors: [N, D]
        """
        if max_features != self.max_features:
            self.detector = KF.SIFTFeature(num_features=max_features).to(DEVICE)
            self.max_features = max_features

        with torch.no_grad():
            lafs, responses, descriptors = self.detector(gray_tensor)

        if lafs.shape[1] == 0:
            return torch.zeros((0, 2), device=DEVICE), None

        # LAFs: [B, N, 2, 3] - center is last column as (x, y)
        keypoints_xy = lafs[0, :, :, 2]  # [N, 2]

        # Flip (x, y) -> (y, x)
        keypoints = torch.flip(keypoints_xy, dims=[1])

        return keypoints, descriptors[0]
