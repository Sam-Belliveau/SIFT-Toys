"""
GPU Utilities
Uses: profiler
Used by: all modules

MPS-compatible GPU operations.
"""

import torch
import torch.nn.functional as F
from profiler import profiler

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def grid_sample_safe(
    input,
    grid,
    mode="bilinear",
):
    """MPS-compatible grid sampling with clamp padding."""
    grid_clamped = torch.clamp(grid, -1.0, 1.0)
    return F.grid_sample(
        input,
        grid_clamped,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )


def draw_points(
    image_tensor,
    points,
    color,
    radius=3,
):
    """Draw circles on image tensor (GPU)."""
    with profiler.section("draw_points"):
        if points is None or points.shape[0] == 0:
            return image_tensor

        _, c, h, w = image_tensor.shape
        result = image_tensor.clone()

        with profiler.section("meshgrid"):
            yy, xx = torch.meshgrid(
                torch.arange(h, device=DEVICE, dtype=torch.float32),
                torch.arange(w, device=DEVICE, dtype=torch.float32),
                indexing="ij",
            )

        with profiler.section("per_point"):
            for i in range(points.shape[0]):
                py, px = points[i]
                if torch.isnan(py) or torch.isnan(px):
                    continue
                dist_sq = (yy - py) ** 2 + (xx - px) ** 2
                mask = dist_sq <= radius**2

                for ch in range(c):
                    result[0, ch] = torch.where(mask, color[ch], result[0, ch])

        return result


def draw_lines(
    image_tensor,
    points1,
    points2,
    color,
    thickness=1,
):
    """Draw lines between point pairs on image tensor (GPU)."""
    with profiler.section("draw_lines"):
        if points1 is None or points2 is None or points1.shape[0] == 0:
            return image_tensor

        _, c, h, w = image_tensor.shape
        result = image_tensor.clone()

        with profiler.section("meshgrid"):
            yy, xx = torch.meshgrid(
                torch.arange(h, device=DEVICE, dtype=torch.float32),
                torch.arange(w, device=DEVICE, dtype=torch.float32),
                indexing="ij",
            )

        with profiler.section("per_line"):
            for i in range(min(points1.shape[0], points2.shape[0])):
                y1, x1 = points1[i]
                y2, x2 = points2[i]

                if (
                    torch.isnan(y1)
                    or torch.isnan(x1)
                    or torch.isnan(y2)
                    or torch.isnan(x2)
                ):
                    continue

                dx = x2 - x1
                dy = y2 - y1
                length_sq = dx * dx + dy * dy

                if length_sq < 1e-6:
                    continue

                t = torch.clamp(((xx - x1) * dx + (yy - y1) * dy) / length_sq, 0, 1)
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy

                dist_sq = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
                mask = dist_sq <= thickness**2

                for ch in range(c):
                    result[0, ch] = torch.where(mask, color[ch], result[0, ch])

        return result
