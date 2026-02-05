"""
Debug Overlay Drawing
Uses: nothing
Used by: main.py
"""

import cv2

# === PARAMETERS ===
RAW_POINT_COLOR = (0, 0, 255)  # Red (BGR)
SMOOTH_POINT_COLOR = (0, 255, 0)  # Green (BGR)
LINE_COLOR = (255, 255, 255)  # White
POINT_RADIUS = 3
LINE_THICKNESS = 1


def draw(image, raw_points, smoothed_points):
    """
    Draw keypoint visualization on image.
    - White lines connecting raw to smoothed
    - Red circles for raw keypoints
    - Green circles for smoothed keypoints
    """
    canvas = image.copy()

    if raw_points is None or smoothed_points is None:
        return canvas

    # Lines from raw to smoothed
    for i in range(len(raw_points)):
        raw_y, raw_x = raw_points[i]
        smooth_y, smooth_x = smoothed_points[i]

        cv2.line(
            canvas,
            (int(raw_x), int(raw_y)),
            (int(smooth_x), int(smooth_y)),
            LINE_COLOR,
            LINE_THICKNESS,
            cv2.LINE_AA,
        )

    # Raw points (red)
    for point in raw_points:
        y, x = point
        cv2.circle(
            canvas,
            (int(x), int(y)),
            POINT_RADIUS,
            RAW_POINT_COLOR,
            -1,
        )

    # Smoothed points (green)
    for point in smoothed_points:
        y, x = point
        cv2.circle(
            canvas,
            (int(x), int(y)),
            POINT_RADIUS,
            SMOOTH_POINT_COLOR,
            -1,
        )

    return canvas
