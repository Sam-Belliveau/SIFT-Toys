"""
Feature Tracker with LRU Cache (GPU)
Uses: nothing
Used by: main.py

Maintains a cache of tracked features across frames.
All operations use torch tensors on GPU.
"""

import torch
import kornia.feature as KF
import time

# === PARAMETERS ===
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MAX_CACHE_SIZE = 200
STALE_THRESHOLD_SEC = 2.0
MATCH_RATIO = 0.75


class FeatureTracker:
    """GPU-accelerated feature tracker."""

    def __init__(self):
        self.positions = None  # [N, 2] tensor
        self.descriptors = None  # [N, D] tensor
        self.last_seen = None  # [N] tensor of timestamps

        # Kornia matcher
        self.matcher = KF.DescriptorMatcher("snn", th=MATCH_RATIO)

    def update(self, keypoints, descriptors, dt, rc_milliseconds):
        """
        Update tracker with new detections.

        Args:
            keypoints: [N, 2] tensor of (y, x) positions
            descriptors: [N, D] tensor of descriptors
            dt: time since last frame (seconds)
            rc_milliseconds: smoothing time constant

        Returns:
            (matched_detected, matched_tracked): both [M, 2] tensors
        """
        current_time = time.time()

        if self.positions is None or self.positions.shape[0] == 0:
            self.positions = keypoints.clone()
            self.descriptors = descriptors.clone()
            self.last_seen = torch.full(
                (keypoints.shape[0],), current_time, device=DEVICE
            )
            return keypoints, keypoints.clone()

        if descriptors is None or keypoints.shape[0] == 0:
            return torch.zeros((0, 2), device=DEVICE), torch.zeros(
                (0, 2), device=DEVICE
            )

        # Match: descriptors [N, D], self.descriptors [M, D]
        with torch.no_grad():
            dists, match_idxs = self.matcher(descriptors, self.descriptors)

        # match_idxs: [N, 2] where each row is (query_idx, train_idx)
        # Filter valid matches (train_idx != -1)
        valid_mask = match_idxs[:, 1] >= 0
        valid_matches = match_idxs[valid_mask]

        if valid_matches.shape[0] == 0:
            # No matches - add all as new
            self.positions = torch.cat([self.positions, keypoints], dim=0)
            self.descriptors = torch.cat([self.descriptors, descriptors], dim=0)
            self.last_seen = torch.cat(
                [
                    self.last_seen,
                    torch.full((keypoints.shape[0],), current_time, device=DEVICE),
                ]
            )
            self.evict_stale(current_time)
            return torch.zeros((0, 2), device=DEVICE), torch.zeros(
                (0, 2), device=DEVICE
            )

        matched_detected = []
        matched_tracked = []
        matched_cache_set = set()

        for query_idx, cache_idx in valid_matches.tolist():
            cache_idx = int(cache_idx)
            if cache_idx in matched_cache_set:
                continue
            matched_cache_set.add(cache_idx)

            # Calculate alpha based on this feature's last seen time
            feature_dt = current_time - self.last_seen[cache_idx].item()
            alpha = self.calculate_alpha(feature_dt, rc_milliseconds)

            new_pos = keypoints[query_idx]
            old_pos = self.positions[cache_idx]
            smoothed = new_pos * alpha + old_pos * (1.0 - alpha)

            self.positions[cache_idx] = smoothed
            self.descriptors[cache_idx] = descriptors[query_idx]
            self.last_seen[cache_idx] = current_time

            matched_detected.append(new_pos)
            matched_tracked.append(smoothed)

        # Add unmatched detections
        matched_query_set = set(valid_matches[:, 0].tolist())
        for i in range(keypoints.shape[0]):
            if i not in matched_query_set:
                self.positions = torch.cat(
                    [self.positions, keypoints[i : i + 1]], dim=0
                )
                self.descriptors = torch.cat(
                    [self.descriptors, descriptors[i : i + 1]], dim=0
                )
                self.last_seen = torch.cat(
                    [self.last_seen, torch.tensor([current_time], device=DEVICE)]
                )

        self.evict_stale(current_time)

        if len(matched_detected) == 0:
            return torch.zeros((0, 2), device=DEVICE), torch.zeros(
                (0, 2), device=DEVICE
            )

        return torch.stack(matched_detected), torch.stack(matched_tracked)

    def evict_stale(self, current_time):
        """Remove features not seen recently."""
        if self.last_seen is None:
            return

        age = current_time - self.last_seen.cpu().numpy()
        keep_mask = torch.from_numpy(age < STALE_THRESHOLD_SEC).to(DEVICE)

        if keep_mask.sum() > MAX_CACHE_SIZE:
            sorted_indices = torch.argsort(-self.last_seen)[:MAX_CACHE_SIZE]
            keep_mask = torch.zeros(
                self.last_seen.shape[0], dtype=torch.bool, device=DEVICE
            )
            keep_mask[sorted_indices] = True

        if keep_mask.sum() < self.positions.shape[0]:
            self.positions = self.positions[keep_mask]
            self.descriptors = self.descriptors[keep_mask]
            self.last_seen = self.last_seen[keep_mask]

    def reset(self):
        self.positions = None
        self.descriptors = None
        self.last_seen = None

    @staticmethod
    def calculate_alpha(dt, rc_milliseconds):
        if rc_milliseconds <= 0:
            return 1.0
        rc_seconds = rc_milliseconds / 1000.0
        return 1.0 - torch.exp(torch.tensor(-dt / rc_seconds)).item()
