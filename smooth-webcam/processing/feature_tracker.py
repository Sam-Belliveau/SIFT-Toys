"""
Feature Tracker (CPU)

Simple ORB feature tracking using cv2.BFMatcher with Hamming distance.
"""

import time
import numpy as np
import cv2
from profiler import profiler

# === PARAMETERS ===
MAX_DISTANCE_PX = 50.0  # Reject matches that jump more than this
STALE_SEC = 0.2
MIN_STABLE_FRAMES = 3


class FeatureTracker:
    def __init__(self):
        self.cache_pos = None
        self.cache_desc = None
        self.cache_time = None
        self.cache_stable = None
        # Hamming distance for binary ORB descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def update(self, keypoints, descriptors, dt, rc_ms):
        """Match new detections to cache, apply smoothing."""
        with profiler.section("update"):
            now = time.time()
            alpha = 1.0 - np.exp(-dt / (rc_ms / 1000.0)) if rc_ms > 0 else 1.0

            # First frame: init cache
            if self.cache_pos is None or len(self.cache_pos) == 0:
                self._init_cache(keypoints, descriptors, now)
                return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

            if descriptors is None or len(keypoints) == 0:
                return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

            # Match with crossCheck (simpler, no ratio test needed)
            with profiler.section("match"):
                try:
                    matches = self.matcher.match(descriptors, self.cache_desc)
                except cv2.error:
                    return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

            # Filter by distance, update cache
            with profiler.section("filter"):
                detected, tracked = [], []
                used_cache = set()

                for m in matches:
                    det_idx, cache_idx = m.queryIdx, m.trainIdx
                    if cache_idx in used_cache:
                        continue

                    new_pos = keypoints[det_idx]
                    old_pos = self.cache_pos[cache_idx]
                    if np.linalg.norm(new_pos - old_pos) > MAX_DISTANCE_PX:
                        continue

                    used_cache.add(cache_idx)

                    smoothed = alpha * new_pos + (1 - alpha) * old_pos
                    self.cache_pos[cache_idx] = smoothed
                    self.cache_desc[cache_idx] = descriptors[det_idx]
                    self.cache_time[cache_idx] = now
                    self.cache_stable[cache_idx] += 1

                    if self.cache_stable[cache_idx] >= MIN_STABLE_FRAMES:
                        detected.append(new_pos)
                        tracked.append(smoothed)

                # Reset stability for unmatched
                for i in range(len(self.cache_pos)):
                    if i not in used_cache:
                        self.cache_stable[i] = 0

            # Add new detections
            with profiler.section("add_new"):
                matched_dets = {m.queryIdx for m in matches if m.trainIdx in used_cache}
                for i in range(len(keypoints)):
                    if i not in matched_dets:
                        self.cache_pos = np.vstack(
                            [self.cache_pos, keypoints[i : i + 1]]
                        )
                        self.cache_desc = np.vstack(
                            [self.cache_desc, descriptors[i : i + 1]]
                        )
                        self.cache_time = np.append(self.cache_time, now)
                        self.cache_stable = np.append(self.cache_stable, 1)

            # Evict stale
            with profiler.section("evict"):
                keep = (now - self.cache_time) < STALE_SEC
                if np.sum(keep) < len(self.cache_pos):
                    self.cache_pos = self.cache_pos[keep]
                    self.cache_desc = self.cache_desc[keep]
                    self.cache_time = self.cache_time[keep]
                    self.cache_stable = self.cache_stable[keep]

            if len(detected) == 0:
                return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

            return np.array(detected, np.float32), np.array(tracked, np.float32)

    def _init_cache(self, keypoints, descriptors, now):
        self.cache_pos = keypoints.copy()
        self.cache_desc = descriptors.copy() if descriptors is not None else None
        self.cache_time = np.full(len(keypoints), now)
        self.cache_stable = np.ones(len(keypoints), dtype=np.int32)

    def reset(self):
        self.cache_pos = None
        self.cache_desc = None
        self.cache_time = None
        self.cache_stable = None
