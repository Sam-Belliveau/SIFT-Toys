"""
Feature Tracker (CPU)
Uses: profiler
Used by: main.py

Tracks features across frames using descriptor matching and temporal smoothing.
"""

import time
import numpy as np
import cv2
from profiler import profiler

# === PARAMETERS ===
MATCH_RATIO = 0.75
STALE_THRESHOLD_SEC = 2.0
MAX_CACHE_SIZE = 256


class FeatureTracker:
    """
    Maintains a cache of tracked features with temporal smoothing.
    """

    def __init__(self):
        self.positions = None
        self.descriptors = None
        self.last_seen = None
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def update(
        self,
        keypoints,
        descriptors,
        dt,
        rc_milliseconds,
    ):
        """
        Update tracker with new detections.

        Args:
            keypoints: Nx2 array of (y, x) positions
            descriptors: NxD array of SIFT descriptors
            dt: Time since last frame (seconds)
            rc_milliseconds: Smoothing time constant

        Returns:
            Tuple of (detected, tracked):
                - detected: positions from current frame that matched
                - tracked: corresponding smoothed positions from cache
        """
        with profiler.section("update"):
            current_time = time.time()
            alpha = self.calculate_alpha(dt, rc_milliseconds)

            if self.positions is None or len(self.positions) == 0:
                self.positions = keypoints.copy()
                self.descriptors = (
                    descriptors.copy() if descriptors is not None else None
                )
                self.last_seen = np.full(len(keypoints), current_time)
                return keypoints, keypoints.copy()

            if descriptors is None or len(keypoints) == 0:
                return np.zeros((0, 2), dtype=np.float32), np.zeros(
                    (0, 2), dtype=np.float32
                )

            with profiler.section("match"):
                matches = self.matcher.knnMatch(descriptors, self.descriptors, k=2)

            with profiler.section("ratio_test"):
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < MATCH_RATIO * n.distance:
                            good_matches.append(m)

            with profiler.section("smooth"):
                matched_cache_indices = set()
                matched_detected = []
                matched_tracked = []

                for m in good_matches:
                    detection_idx = m.queryIdx
                    cache_idx = m.trainIdx

                    if cache_idx in matched_cache_indices:
                        continue

                    matched_cache_indices.add(cache_idx)

                    new_pos = keypoints[detection_idx]
                    old_pos = self.positions[cache_idx]
                    smoothed = (new_pos * alpha) + (old_pos * (1.0 - alpha))

                    self.positions[cache_idx] = smoothed
                    self.descriptors[cache_idx] = descriptors[detection_idx]
                    self.last_seen[cache_idx] = current_time

                    matched_detected.append(new_pos)
                    matched_tracked.append(smoothed)

            with profiler.section("add_new"):
                for i in range(len(keypoints)):
                    was_matched = any(m.queryIdx == i for m in good_matches)
                    if not was_matched:
                        self.positions = np.vstack(
                            [self.positions, keypoints[i : i + 1]]
                        )
                        self.descriptors = np.vstack(
                            [self.descriptors, descriptors[i : i + 1]]
                        )
                        self.last_seen = np.append(self.last_seen, current_time)

            with profiler.section("evict"):
                self.evict_stale(current_time)

            if len(matched_detected) == 0:
                return np.zeros((0, 2), dtype=np.float32), np.zeros(
                    (0, 2), dtype=np.float32
                )

            return np.array(matched_detected, dtype=np.float32), np.array(
                matched_tracked, dtype=np.float32
            )

    def evict_stale(self, current_time):
        """Remove features not seen recently."""
        if self.last_seen is None:
            return

        age = current_time - self.last_seen
        keep_mask = age < STALE_THRESHOLD_SEC

        if np.sum(keep_mask) > MAX_CACHE_SIZE:
            sorted_indices = np.argsort(-self.last_seen)
            keep_indices = sorted_indices[:MAX_CACHE_SIZE]
            keep_mask = np.zeros(len(self.last_seen), dtype=bool)
            keep_mask[keep_indices] = True

        if np.sum(keep_mask) < len(self.positions):
            self.positions = self.positions[keep_mask]
            self.descriptors = self.descriptors[keep_mask]
            self.last_seen = self.last_seen[keep_mask]

    def reset(self):
        """Clear all cached features."""
        self.positions = None
        self.descriptors = None
        self.last_seen = None

    @staticmethod
    def calculate_alpha(dt, rc_milliseconds):
        """Calculate smoothing alpha from time constant."""
        if rc_milliseconds <= 0:
            return 1.0
        rc_seconds = rc_milliseconds / 1000.0
        return 1.0 - np.exp(-dt / rc_seconds)
