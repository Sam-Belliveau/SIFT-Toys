"""
Trackbar Interface
Uses: nothing
Used by: main.py
"""

import cv2

# === PARAMETERS ===
WINDOW_NAME = "RBF Warp"
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

MAX_FEATURES_DEFAULT = 500
MAX_FEATURES_MAX = 1000
MIN_FEATURES = 4

FILTER_RC_DEFAULT = 500
FILTER_RC_MAX = 2000


def nothing(x):
    pass


def setup():
    """Create window and trackbars."""
    # WINDOW_NORMAL allows resizing, WINDOW_KEEPRATIO maintains aspect
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    cv2.createTrackbar(
        "Max Features",
        WINDOW_NAME,
        MAX_FEATURES_DEFAULT,
        MAX_FEATURES_MAX,
        nothing,
    )

    cv2.createTrackbar(
        "Filter RC (ms)",
        WINDOW_NAME,
        FILTER_RC_DEFAULT,
        FILTER_RC_MAX,
        nothing,
    )


def read_params():
    """Read current trackbar values. Returns (max_features, rc_ms)."""
    max_features = cv2.getTrackbarPos("Max Features", WINDOW_NAME)
    rc_ms = cv2.getTrackbarPos("Filter RC (ms)", WINDOW_NAME)

    if max_features < MIN_FEATURES:
        max_features = MIN_FEATURES

    return max_features, rc_ms
