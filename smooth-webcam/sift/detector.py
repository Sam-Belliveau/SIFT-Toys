"""
SIFT Feature Detector (CPU)
Uses: profiler
Used by: main.py

OpenCV SIFT - fast, optimized C++.
"""

import cv2
import numpy as np
from profiler import profiler

# === PARAMETERS ===
DEFAULT_MAX_FEATURES = 64


class SIFTDetector:
    """CPU-based SIFT detector using OpenCV."""

    def __init__(self):
        self.detector = cv2.SIFT_create(nfeatures=DEFAULT_MAX_FEATURES)
        self.max_features = DEFAULT_MAX_FEATURES

    def detect(
        self,
        gray,
        max_features=DEFAULT_MAX_FEATURES,
    ):
        """
        Extract SIFT keypoints and descriptors.

        Args:
            gray: HxW uint8 numpy array
            max_features: Maximum number of features to detect

        Returns:
            (keypoints, descriptors):
                - keypoints: Nx2 float32 array as (y, x)
                - descriptors: NxD float32 array
        """
        with profiler.section("rebuild_detector"):
            if max_features != self.max_features:
                self.detector = cv2.SIFT_create(nfeatures=max_features)
                self.max_features = max_features

        with profiler.section("opencv_detect"):
            kps, descriptors = self.detector.detectAndCompute(gray, None)

        with profiler.section("convert_keypoints"):
            if kps is None or len(kps) == 0:
                return np.zeros((0, 2), dtype=np.float32), None

            # OpenCV keypoints are (x, y), convert to (y, x)
            keypoints = np.array(
                [[kp.pt[1], kp.pt[0]] for kp in kps],
                dtype=np.float32,
            )

        return keypoints, descriptors
