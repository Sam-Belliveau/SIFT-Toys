"""
Feature Tracker (CPU)

Lucas-Kanade optical flow tracking with exponential smoothing.
"""

import numpy as np
import cv2
from profiler import profiler

# === PARAMETERS ===
MAX_CORNERS = 500
CORNER_QUALITY = 0.01
CORNER_MIN_DIST = 15
BLOCK_SIZE = 7

LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01,
)

BACKTRACK_THRESHOLD = 1.0
REDETECT_INTERVAL = 10
MIN_TRACK_COUNT = 20


class FeatureTracker:
    def __init__(self):
        self.prev_gray = None
        self.raw_points = None
        self.smooth_points = None
        self.frame_count = 0

    def update(
        self,
        gray,
        dt,
        rc_ms,
        max_features=MAX_CORNERS,
    ):
        alpha = 1.0 - np.exp(-dt / (rc_ms / 1000.0)) if rc_ms > 0 else 1.0

        if self.prev_gray is None:
            with profiler.section("detect"):
                self._detect(gray, max_features)
            self.prev_gray = gray
            return (
                np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
            )

        self.frame_count += 1

        with profiler.section("optical_flow"):
            raw, smooth = self._track(gray, alpha)

        needs_redetect = (
            self.frame_count % REDETECT_INTERVAL == 0
            or len(self.raw_points) < MIN_TRACK_COUNT
        )

        if needs_redetect:
            with profiler.section("redetect"):
                self._add_new_points(gray, max_features)

        self.prev_gray = gray
        return raw, smooth

    def _detect(
        self,
        gray,
        max_features,
    ):
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_features,
            qualityLevel=CORNER_QUALITY,
            minDistance=CORNER_MIN_DIST,
            blockSize=BLOCK_SIZE,
        )

        if corners is None:
            self.raw_points = np.zeros((0, 2), np.float32)
            self.smooth_points = np.zeros((0, 2), np.float32)
            return

        pts = corners.reshape(-1, 2)
        # Convert (x, y) → (y, x) for consistency
        pts = pts[:, ::-1].copy().astype(np.float32)
        self.raw_points = pts
        self.smooth_points = pts.copy()

    def _track(
        self,
        gray,
        alpha,
    ):
        if self.raw_points is None or len(self.raw_points) == 0:
            return (
                np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
            )

        # Points for LK need (x, y) shape (N, 1, 2)
        prev_pts = self.raw_points[:, ::-1].reshape(-1, 1, 2).astype(np.float32)

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

        if not np.any(valid):
            self.raw_points = np.zeros((0, 2), np.float32)
            self.smooth_points = np.zeros((0, 2), np.float32)
            return (
                np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
            )

        with profiler.section("smooth"):
            # Convert tracked points back to (y, x)
            new_raw = next_pts.reshape(-1, 2)[valid][:, ::-1].copy().astype(np.float32)
            old_smooth = self.smooth_points[valid]

            new_smooth = alpha * new_raw + (1.0 - alpha) * old_smooth

            self.raw_points = new_raw
            self.smooth_points = new_smooth

        return self.raw_points.copy(), self.smooth_points.copy()

    def _add_new_points(
        self,
        gray,
        max_features,
    ):
        existing_count = len(self.raw_points) if self.raw_points is not None else 0
        needed = max_features - existing_count

        if needed <= 0:
            return

        mask = np.full(gray.shape[:2], 255, dtype=np.uint8)

        if self.raw_points is not None and len(self.raw_points) > 0:
            for pt in self.raw_points:
                y, x = int(pt[0]), int(pt[1])
                cv2.circle(mask, (x, y), int(CORNER_MIN_DIST), 0, -1)

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=needed,
            qualityLevel=CORNER_QUALITY,
            minDistance=CORNER_MIN_DIST,
            blockSize=BLOCK_SIZE,
            mask=mask,
        )

        if corners is None:
            return

        new_pts = corners.reshape(-1, 2)[:, ::-1].copy().astype(np.float32)

        if self.raw_points is not None and len(self.raw_points) > 0:
            self.raw_points = np.vstack([self.raw_points, new_pts])
            self.smooth_points = np.vstack([self.smooth_points, new_pts])
        else:
            self.raw_points = new_pts
            self.smooth_points = new_pts.copy()

    def reset(self):
        self.prev_gray = None
        self.raw_points = None
        self.smooth_points = None
        self.frame_count = 0
