"""
ORB Feature Detector (CPU)

Fast binary descriptor with spatial distribution.
"""

import cv2
import numpy as np
from profiler import profiler

DEFAULT_MAX_FEATURES = 500
MIN_SPACING_PX = 10  # Minimum distance between features


class ORBDetector:
    """CPU-based ORB detector with spatial filtering to prevent clumping."""

    def __init__(self):
        # Detect many features, then filter spatially
        self.detector = cv2.ORB_create(nfeatures=DEFAULT_MAX_FEATURES * 2)
        self.max_features = DEFAULT_MAX_FEATURES

    def detect(self, gray, max_features=DEFAULT_MAX_FEATURES):
        """
        Extract ORB keypoints with spatial distribution.

        Returns:
            (keypoints, descriptors):
                - keypoints: Nx2 float32 array as (y, x)
                - descriptors: NxD uint8 array (binary)
        """
        with profiler.section("rebuild_detector"):
            if max_features != self.max_features:
                self.detector = cv2.ORB_create(nfeatures=max_features * 2)
                self.max_features = max_features

        with profiler.section("opencv_detect"):
            kps, descriptors = self.detector.detectAndCompute(gray, None)

        with profiler.section("convert_keypoints"):
            if kps is None or len(kps) == 0:
                return np.zeros((0, 2), dtype=np.float32), None

            # Sort by response (strongest first)
            indices = np.argsort([-kp.response for kp in kps])

            # Spatial filtering: keep only well-separated points
            kept = []
            kept_positions = []

            for idx in indices:
                kp = kps[idx]
                pos = np.array([kp.pt[1], kp.pt[0]])  # (y, x)

                # Check distance to all kept points
                if len(kept_positions) > 0:
                    dists = np.linalg.norm(np.array(kept_positions) - pos, axis=1)
                    if np.min(dists) < MIN_SPACING_PX:
                        continue

                kept.append(idx)
                kept_positions.append(pos)

                if len(kept) >= max_features:
                    break

            if len(kept) == 0:
                return np.zeros((0, 2), dtype=np.float32), None

            keypoints = np.array(kept_positions, dtype=np.float32)
            descriptors = descriptors[kept]

        return keypoints, descriptors
