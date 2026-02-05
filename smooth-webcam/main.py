"""
Smooth Webcam - GPU Accelerated SIFT Feature Tracking

Main orchestrator.

CPU transfers (justified):
  1. Camera capture -> tensor: Unavoidable, camera outputs numpy
  2. Tensor -> display: Unavoidable, cv2.imshow needs numpy
  3. RBF interpolation: scipy has no GPU support
"""

import time
import torch
import cv2

from frontend import video_capture, interface, display
from sift.detector import SIFTDetector
from processing import grayscale, image_warp, gpu_utils
from processing.feature_tracker import FeatureTracker
from profiler import profiler

# === PARAMETERS ===
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MIN_FEATURES = 4

# Colors for GPU drawing (RGB, 0-1)
COLOR_RED = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
COLOR_GREEN = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
COLOR_WHITE = torch.tensor([1.0, 1.0, 1.0], device=DEVICE)


def frame_to_tensor(frame):
    """CPU->GPU transfer: Unavoidable, camera outputs numpy."""
    with profiler.section("bgr_to_rgb"):
        rgb = frame[:, :, ::-1].copy()
    with profiler.section("to_device"):
        tensor = torch.from_numpy(rgb).float().to(DEVICE) / 255.0
    with profiler.section("permute"):
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_frame(tensor):
    """GPU->CPU transfer: Unavoidable, cv2.imshow needs numpy."""
    with profiler.section("permute"):
        rgb = tensor[0].permute(1, 2, 0)
    with profiler.section("to_cpu"):
        rgb = rgb.cpu().numpy()
    with profiler.section("to_uint8"):
        rgb = (rgb * 255).astype("uint8")
    with profiler.section("rgb_to_bgr"):
        bgr = rgb[:, :, ::-1].copy()
    return bgr


def main():
    print(f"Using device: {DEVICE}")

    camera = video_capture.get_stream()
    sift = SIFTDetector()
    tracker = FeatureTracker()

    interface.setup()

    last_time = time.time()

    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            with profiler.section("capture"):
                frame = video_capture.get_frame(camera)
                if frame is None:
                    break

            max_features, rc_ms = interface.read_params()

            with profiler.section("to_tensor"):
                img_tensor = frame_to_tensor(frame)

            with profiler.section("grayscale"):
                gray_tensor = grayscale.convert(img_tensor)

            with profiler.section("sift"):
                keypoints, descriptors = sift.detect(gray_tensor, max_features)

            if keypoints.shape[0] < MIN_FEATURES:
                output = tensor_to_frame(img_tensor)
                display.show(output)
                profiler.report()
                continue

            with profiler.section("tracking"):
                detected, tracked = tracker.update(keypoints, descriptors, dt, rc_ms)

            if detected.shape[0] < MIN_FEATURES:
                output = tensor_to_frame(img_tensor)
                display.show(output)
                profiler.report()
                continue

            with profiler.section("warp"):
                warped_tensor = image_warp.warp_image(img_tensor, detected, tracked)

            with profiler.section("draw"):
                output_tensor = gpu_utils.draw_lines(
                    warped_tensor, detected, tracked, COLOR_WHITE
                )
                output_tensor = gpu_utils.draw_points(
                    output_tensor, detected, COLOR_RED, radius=3
                )
                output_tensor = gpu_utils.draw_points(
                    output_tensor, tracked, COLOR_GREEN, radius=3
                )

            with profiler.section("to_frame"):
                output = tensor_to_frame(output_tensor)

            with profiler.section("display"):
                if not display.show(output):
                    break

            profiler.report()

    finally:
        video_capture.release(camera)
        display.cleanup()


if __name__ == "__main__":
    main()
