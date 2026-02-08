"""
Feature Tracker (CPU)

Dense hexagonal grid optical flow with exponential smoothing.
Grid points drift back to their original positions via EMA decay.
"""

import numpy as np
import cv2
from profiler import profiler

# === PARAMETERS ===
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01,
)

BACKTRACK_THRESHOLD = 1.0


def generate_hexagonal_grid(height, width, n_samples):
    """Generate evenly-spaced hexagonal grid points covering the image."""
    ratio = (width / height) * np.sqrt(3) / 2

    sw = int(np.ceil(np.sqrt(n_samples * ratio)))
    sh = int(np.round(np.sqrt(n_samples / ratio)))

    xs = np.round(np.linspace(0, width - 1, (2 * sw - 1) + 2, endpoint=True)[1:-1])
    xs_even = xs[0::2]
    xs_odd = xs[1::2]

    ys = np.round(np.linspace(0, height - 1, sh + 2, endpoint=True)[1:-1])

    xi = []
    yi = []
    for i, y in enumerate(ys):
        row_xs = xs_even if i % 2 == 0 else xs_odd
        xi.extend(row_xs)
        yi.extend([y] * len(row_xs))

    # Return as (N, 2) array in (y, x) format
    return np.column_stack([yi, xi]).astype(np.float32)


class FeatureTracker:
    def __init__(self):
        self.prev_gray = None
        self.grid_points = (
            None  # original hex grid positions (y, x) — fixed destination
        )
        self.current_points = None  # LK-tracked, decaying toward grid (y, x) — source
        self.n_samples = None
        self.shape = None

    def _init_grid(self, gray, n_samples):
        h, w = gray.shape[:2]
        self.grid_points = generate_hexagonal_grid(h, w, n_samples)
        self.current_points = self.grid_points.copy()
        self.n_samples = n_samples
        self.shape = gray.shape[:2]

    def update(
        self,
        gray,
        dt,
        rc_ms,
        n_samples=500,
    ):
        alpha = 1.0 - np.exp(-dt / (rc_ms / 1000.0)) if rc_ms > 0 else 1.0

        needs_reinit = (
            self.prev_gray is None
            or self.shape != gray.shape[:2]
            or self.n_samples != n_samples
        )

        if needs_reinit:
            with profiler.section("detect"):
                self._init_grid(gray, n_samples)
            self.prev_gray = gray
            return (
                np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
            )

        with profiler.section("optical_flow"):
            self._track(gray, alpha)

        self.prev_gray = gray
        return self.current_points.copy(), self.grid_points.copy()

    def _track(self, gray, alpha):
        # Points for LK need (x, y) shape (N, 1, 2)
        prev_pts = self.current_points[:, ::-1].reshape(-1, 1, 2).astype(np.float32)

        with profiler.section("forward"):
            next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                prev_pts,
                None,
                winSize=LK_WIN_SIZE,
                maxLevel=LK_MAX_LEVEL,
                criteria=LK_CRITERIA,
            )

        with profiler.section("backward"):
            back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self.prev_gray,
                next_pts,
                None,
                winSize=LK_WIN_SIZE,
                maxLevel=LK_MAX_LEVEL,
                criteria=LK_CRITERIA,
            )

        with profiler.section("validate"):
            fb_error = np.linalg.norm(
                prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2),
                axis=1,
            )
            valid = (
                (status_fwd.ravel() == 1)
                & (status_bwd.ravel() == 1)
                & (fb_error < BACKTRACK_THRESHOLD)
            )

        with profiler.section("decay"):
            # Convert tracked points back to (y, x)
            tracked = next_pts.reshape(-1, 2)[:, ::-1].copy().astype(np.float32)

            # Valid points: follow LK result
            # Invalid points: snap back to grid origin
            self.current_points[valid] = tracked[valid]
            self.current_points[~valid] = self.grid_points[~valid]

            # Decay toward grid — when motion stops, points relax back
            self.current_points = (
                1.0 - alpha
            ) * self.current_points + alpha * self.grid_points

    def reset(self):
        self.prev_gray = None
        self.grid_points = None
        self.current_points = None
        self.n_samples = None
        self.shape = None
