"""
Point Matching Across Frames
Uses: nothing
Used by: main.py

Matches current keypoints to previous keypoints using SIFT descriptors.
"""

import cv2
import numpy as np

# === PARAMETERS ===
MATCH_RATIO = 0.75  # Lowe's ratio test threshold


def match_points(
    current_keypoints,
    current_descriptors,
    previous_keypoints,
    previous_descriptors,
):
    """
    Match current features to previous features using descriptor matching.

    Returns:
        - matched_current: keypoints from current frame that have matches
        - matched_previous: corresponding keypoints from previous frame

    If no previous data, returns (current_keypoints, current_keypoints).
    """
    if previous_keypoints is None or previous_descriptors is None:
        return current_keypoints, current_keypoints.copy()

    if len(current_keypoints) == 0 or len(previous_keypoints) == 0:
        return current_keypoints, current_keypoints.copy()

    if current_descriptors is None or previous_descriptors is None:
        return current_keypoints, current_keypoints.copy()

    # Use FLANN matcher for speed
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # Match current -> previous
    matches = matcher.knnMatch(current_descriptors, previous_descriptors, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < MATCH_RATIO * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        return current_keypoints, current_keypoints.copy()

    # Extract matched points
    matched_current = np.array(
        [current_keypoints[m.queryIdx] for m in good_matches],
        dtype=np.float32,
    )
    matched_previous = np.array(
        [previous_keypoints[m.trainIdx] for m in good_matches],
        dtype=np.float32,
    )

    return matched_current, matched_previous
