"""
Feature Tracker (Stateful)
Uses: nothing
Used by: main.py

Tracks SIFT features across frames, smoothing their positions over time.
"""

import numpy as np


class FeatureTracker:
    """Maintains tracked feature positions across frames."""

    def __init__(self):
        self.tracked = None

    def update(self, target_positions, dt, rc_milliseconds):
        """
        Update tracked positions toward new targets.

        Args:
            target_positions: New feature positions to track toward
            dt: Time delta since last update (seconds)
            rc_milliseconds: Tracking time constant (higher = slower tracking)

        Returns:
            Current tracked positions
        """
        alpha = self.calculate_alpha(dt, rc_milliseconds)

        if self.tracked is None:
            self.tracked = target_positions.copy()
        else:
            self.tracked = (target_positions * alpha) + (self.tracked * (1.0 - alpha))

        return self.tracked

    def reset(self):
        """Clear tracked state."""
        self.tracked = None

    @staticmethod
    def calculate_alpha(dt, rc_milliseconds):
        """Calculate tracking rate from time constant."""
        if rc_milliseconds <= 0:
            return 1.0
        rc_seconds = rc_milliseconds / 1000.0
        return 1.0 - np.exp(-dt / rc_seconds)
