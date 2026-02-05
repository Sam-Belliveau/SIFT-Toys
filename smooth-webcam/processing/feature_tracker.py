"""
Feature Tracker with LRU Cache (Stateful)
Uses: nothing
Used by: main.py

Maintains a cache of tracked features across frames.
Features are matched by descriptor, positions are smoothed over time,
and stale features are evicted using LRU policy.
"""

import cv2
import numpy as np
import time

# === PARAMETERS ===
MAX_CACHE_SIZE = 200  # Maximum features to track
STALE_THRESHOLD_SEC = 2.0  # Evict features not seen for this long
MATCH_RATIO = 0.75  # Lowe's ratio test threshold


class FeatureTracker:
    """
    Maintains a cache of tracked features with temporal smoothing.

    Each cached feature stores:
        - position: smoothed (y, x) position
        - descriptor: SIFT descriptor for matching
        - last_seen: timestamp of last update
    """

    def __init__(self):
        self.positions = None  # Nx2 array of smoothed positions
        self.descriptors = None  # NxD array of descriptors
        self.last_seen = None  # N array of timestamps

        # Brute force matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def update(self, keypoints, descriptors, dt, rc_milliseconds):
        """
        Update tracker with new detections.

        Args:
            keypoints: Nx2 array of detected (y, x) positions
            descriptors: NxD array of SIFT descriptors
            dt: Time since last frame (seconds)
            rc_milliseconds: Smoothing time constant

        Returns:
            Tuple of (matched_detected, matched_tracked):
                - matched_detected: positions from current frame that matched
                - matched_tracked: corresponding smoothed positions from cache
        """
        current_time = time.time()

        if self.positions is None or len(self.positions) == 0:
            # First frame - initialize cache
            self.positions = keypoints.copy()
            self.descriptors = descriptors.copy()
            self.last_seen = np.full(len(keypoints), current_time)
            return keypoints, keypoints.copy()

        if descriptors is None or len(keypoints) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))

        # Match new detections to cached features
        matches = self.matcher.knnMatch(descriptors, self.descriptors, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < MATCH_RATIO * n.distance:
                    good_matches.append(m)

        # Track which cache entries were matched
        matched_cache_indices = set()
        matched_detected = []
        matched_tracked = []

        for m in good_matches:
            detection_idx = m.queryIdx
            cache_idx = m.trainIdx

            if cache_idx in matched_cache_indices:
                continue  # Already matched this cache entry

            matched_cache_indices.add(cache_idx)

            # Smooth position toward new detection
            new_pos = keypoints[detection_idx]
            old_pos = self.positions[cache_idx]
            dt = current_time - self.last_seen[cache_idx]
            alpha = self.calculate_alpha(dt, rc_milliseconds)
            smoothed = (new_pos * alpha) + (old_pos * (1.0 - alpha))

            self.positions[cache_idx] = smoothed
            self.descriptors[cache_idx] = descriptors[detection_idx]
            self.last_seen[cache_idx] = current_time

            matched_detected.append(new_pos)
            matched_tracked.append(smoothed)

        # Add unmatched detections as new cache entries
        for i in range(len(keypoints)):
            if not any(
                m.queryIdx == i
                for m in good_matches
                if m.trainIdx in matched_cache_indices or True
            ):
                # Check if this detection was matched
                was_matched = any(m.queryIdx == i for m in good_matches)
                if not was_matched:
                    self.positions = np.vstack([self.positions, keypoints[i : i + 1]])
                    self.descriptors = np.vstack(
                        [self.descriptors, descriptors[i : i + 1]]
                    )
                    self.last_seen = np.append(self.last_seen, current_time)

        # Evict stale entries (LRU)
        self.evict_stale(current_time)

        if len(matched_detected) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))

        return np.array(matched_detected), np.array(matched_tracked)

    def evict_stale(self, current_time):
        """Remove features not seen recently."""
        if self.last_seen is None:
            return

        # Keep features seen within threshold, or if cache under limit
        age = current_time - self.last_seen
        keep_mask = age < STALE_THRESHOLD_SEC

        # Also enforce max cache size by keeping most recent
        if np.sum(keep_mask) > MAX_CACHE_SIZE:
            # Keep only the MAX_CACHE_SIZE most recently seen
            sorted_indices = np.argsort(-self.last_seen)  # Most recent first
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

    def get_cache_size(self):
        """Return current number of cached features."""
        return 0 if self.positions is None else len(self.positions)

    @staticmethod
    def calculate_alpha(dt, rc_milliseconds):
        """Calculate smoothing alpha from time constant."""
        if rc_milliseconds <= 0:
            return 1.0
        rc_seconds = rc_milliseconds / 1000.0
        return 1.0 - np.exp(-dt / rc_seconds)
