"""
Low-Pass Filter (Stateful)
Uses: nothing
Used by: main.py

Exponential smoothing filter for temporal point stabilization.
"""

import numpy as np


class LowPassFilter:
    """Maintains smoothed values across frames."""

    def __init__(self):
        self._smoothed = None

    def update(self, target_values, dt, rc_milliseconds):
        """
        Update filter with new target values.

        Args:
            target_values: New values to smooth towards
            dt: Time delta since last update (seconds)
            rc_milliseconds: Filter time constant (higher = more smoothing)

        Returns:
            Smoothed values
        """
        alpha = self._calculate_alpha(dt, rc_milliseconds)

        if self._smoothed is None:
            self._smoothed = target_values.copy()
        else:
            self._smoothed = (target_values * alpha) + (self._smoothed * (1.0 - alpha))

        return self._smoothed

    def reset(self):
        """Clear filter state."""
        self._smoothed = None

    def get_smoothed(self):
        """Get current smoothed values (may be None)."""
        return self._smoothed

    @staticmethod
    def _calculate_alpha(dt, rc_milliseconds):
        """Calculate smoothing alpha from time constant."""
        if rc_milliseconds <= 0:
            return 1.0
        rc_seconds = rc_milliseconds / 1000.0
        return 1.0 - np.exp(-dt / rc_seconds)
