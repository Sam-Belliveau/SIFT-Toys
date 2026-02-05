"""
SIFT Feature Detector (Stateful)
Uses: nothing
Used by: main.py

Keeps detector instance across frames to avoid recreation overhead.
"""

import cv2
import numpy as np

# === PARAMETERS ===
DEFAULT_MAX_FEATURES = 64


class SIFTDetector:
    """Reusable SIFT detector that persists across frames."""

    def __init__(self):
        self.detector = None
        self.max_features = None

    def detect(self, gray_image, max_features=DEFAULT_MAX_FEATURES):
        """
        Extract SIFT keypoints and descriptors from grayscale image.
        Returns (keypoints, descriptors):
            - keypoints: Nx2 array of (y, x) coordinates
            - descriptors: NxD array of feature descriptors
        """
        # Recreate detector only if max_features changed
        if self.detector is None or self.max_features != max_features:
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.max_features = max_features

        keypoints_raw, descriptors = self.detector.detectAndCompute(gray_image, None)

        if not keypoints_raw:
            return np.zeros((0, 2)), None

        # OpenCV returns (x, y), we flip to (y, x)
        keypoints = np.array(
            [[kp.pt[1], kp.pt[0]] for kp in keypoints_raw],
            dtype=np.float32,
        )

        return keypoints, descriptors
