"""
Drawing Utilities (CPU)
Uses: profiler
Used by: main.py

OpenCV-based drawing functions.
"""

import cv2
import numpy as np
from profiler import profiler


def draw_points(
    image,
    points,
    color,
    radius=3,
):
    """
    Draw circles at points.

    Args:
        image: HxWxC uint8 numpy array (modified in place)
        points: Nx2 array (y, x)
        color: BGR tuple (b, g, r)
        radius: Circle radius in pixels
    """
    with profiler.section("draw_points"):
        if points is None or len(points) == 0:
            return image

        result = image.copy()
        for i in range(len(points)):
            y, x = points[i]
            if np.isnan(y) or np.isnan(x):
                continue
            cv2.circle(
                result,
                (int(x), int(y)),
                radius,
                color,
                -1,
            )
        return result


def draw_lines(
    image,
    points1,
    points2,
    color,
    thickness=1,
):
    """
    Draw lines between point pairs.

    Args:
        image: HxWxC uint8 numpy array
        points1: Nx2 array (y, x) - start points
        points2: Nx2 array (y, x) - end points
        color: BGR tuple (b, g, r)
        thickness: Line thickness
    """
    with profiler.section("draw_lines"):
        if points1 is None or points2 is None or len(points1) == 0:
            return image

        result = image.copy()
        for i in range(min(len(points1), len(points2))):
            y1, x1 = points1[i]
            y2, x2 = points2[i]
            if np.isnan(y1) or np.isnan(x1) or np.isnan(y2) or np.isnan(x2):
                continue
            cv2.line(
                result,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness,
            )
        return result
