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
        self._detector = None
        self._max_features = None

    def detect(self, gray_image, max_features=DEFAULT_MAX_FEATURES):
        """
        Extract SIFT keypoints from grayscale image.
        Returns Nx2 array of (y, x) coordinates.
        """
        # Recreate detector only if max_features changed
        if self._detector is None or self._max_features != max_features:
            self._detector = cv2.SIFT_create(nfeatures=max_features)
            self._max_features = max_features

        keypoints, _ = self._detector.detectAndCompute(gray_image, None)

        if not keypoints:
            return np.zeros((0, 2))

        # OpenCV returns (x, y), we flip to (y, x)
        return np.array(
            [[kp.pt[1], kp.pt[0]] for kp in keypoints],
            dtype=np.float32,
        )
