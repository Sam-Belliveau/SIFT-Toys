"""
Drawing Utilities (CPU)
Uses: profiler
Used by: main.py

OpenCV-based drawing functions.
"""

import cv2
import numpy as np
from profiler import profiler
from params import params


def draw_points(
    image,
    points,
    color,
    radius=3,
):
    with profiler.section("draw_points"):
        if points is None or len(points) == 0:
            return image

        alpha = params["point_opacity"] / 100.0

        overlay = image.copy()
        for i in range(len(points)):
            y, x = points[i]
            if np.isnan(y) or np.isnan(x):
                continue
            cv2.circle(
                overlay,
                (int(x), int(y)),
                radius,
                color,
                -1,
            )
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_lines(
    image,
    points1,
    points2,
    color,
    thickness=1,
):
    with profiler.section("draw_lines"):
        if points1 is None or points2 is None or len(points1) == 0:
            return image

        alpha = params["point_opacity"] / 100.0

        overlay = image.copy()
        for i in range(min(len(points1), len(points2))):
            y1, x1 = points1[i]
            y2, x2 = points2[i]
            if np.isnan(y1) or np.isnan(x1) or np.isnan(y2) or np.isnan(x2):
                continue
            cv2.line(
                overlay,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness,
            )
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
