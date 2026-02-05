"""
Smooth Webcam - GPU Accelerated SIFT Feature Tracking

Main orchestrator.

CPU transfers (justified):
  1. Camera capture -> tensor: Unavoidable, camera outputs numpy
  2. Tensor -> display: Unavoidable, cv2.imshow needs numpy
  3. RBF interpolation: scipy has no GPU support, but only points transfer (small)

Everything else stays on GPU.
"""

import time
import torch
import cv2

from frontend import video_capture, interface, display
from sift.detector import SIFTDetector
from processing import grayscale, image_warp, gpu_utils
from processing.feature_tracker import FeatureTracker

# === PARAMETERS ===
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MIN_FEATURES = 4

# Colors for GPU drawing (RGB, 0-1)
COLOR_RED = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
COLOR_GREEN = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
COLOR_WHITE = torch.tensor([1.0, 1.0, 1.0], device=DEVICE)


def frame_to_tensor(frame):
    """
    Capture frame to GPU tensor.
    CPU->GPU transfer: Unavoidable, camera outputs numpy.
    """
    rgb = frame[:, :, ::-1].copy()
    tensor = torch.from_numpy(rgb).float().to(DEVICE) / 255.0
    return tensor.permute(2, 0, 1).unsqueeze(0)


def tensor_to_frame(tensor):
    """
    GPU tensor to display frame.
    GPU->CPU transfer: Unavoidable, cv2.imshow needs numpy.
    """
    rgb = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return rgb[:, :, ::-1].copy()


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

            # CPU->GPU: camera capture (unavoidable)
            frame = video_capture.get_frame(camera)
            if frame is None:
                break

            max_features, rc_ms = interface.read_params()

            img_tensor = frame_to_tensor(frame)
            gray_tensor = grayscale.convert(img_tensor)

            # Feature detection (GPU)
            keypoints, descriptors = sift.detect(gray_tensor, max_features)

            if keypoints.shape[0] < MIN_FEATURES:
                output = tensor_to_frame(img_tensor)
                display.show(output)
                continue

            # Feature tracking (GPU)
            detected, tracked = tracker.update(keypoints, descriptors, dt, rc_ms)

            if detected.shape[0] < MIN_FEATURES:
                output = tensor_to_frame(img_tensor)
                display.show(output)
                continue

            # Image warp (GPU, with CPU hop for RBF only)
            warped_tensor = image_warp.warp_image(img_tensor, detected, tracked)

            # Draw overlay (GPU)
            output_tensor = gpu_utils.draw_lines(
                warped_tensor, detected, tracked, COLOR_WHITE
            )
            output_tensor = gpu_utils.draw_points(
                output_tensor, detected, COLOR_RED, radius=3
            )
            output_tensor = gpu_utils.draw_points(
                output_tensor, tracked, COLOR_GREEN, radius=3
            )

            # GPU->CPU: display (unavoidable)
            output = tensor_to_frame(output_tensor)

            if not display.show(output):
                break

    finally:
        video_capture.release(camera)
        display.cleanup()


if __name__ == "__main__":
    main()
