"""
Point Matching Across Frames
Uses: nothing
Used by: main.py

Matches current keypoints to previous keypoints for tracking.
"""

import numpy as np
from scipy import optimize


def distance_matrix(points_a, points_b):
    """Compute pairwise Euclidean distances between two point sets."""
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def match_points(current_points, previous_points):
    """
    Match current keypoints to previous keypoints.

    Returns matched previous points in same order as current,
    or current points if no previous history.
    """
    if previous_points is None:
        return current_points

    if len(current_points) == 0:
        return previous_points

    cost_matrix = distance_matrix(current_points, previous_points)

    row_indices, col_indices = optimize.linear_sum_assignment(cost_matrix)

    matched = np.zeros_like(current_points)
    matched[:] = current_points[:]
    matched[row_indices] = previous_points[col_indices]

    return matched
