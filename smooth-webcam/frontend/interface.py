"""
PyQt6 Interface

Window with video display and parameter sliders.
Sliders write directly to the global params registry.
"""

import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

from params import params

# === PARAMETERS ===
WINDOW_TITLE = "Smooth Webcam"
WINDOW_WIDTH = 860
WINDOW_HEIGHT = 580

STYLE = """
    QMainWindow {
        background-color: #1e1e2e;
    }
    QLabel {
        color: #cdd6f4;
        font-size: 12px;
    }
    QLabel#video {
        background-color: #181825;
        border-radius: 6px;
    }
    QSlider::groove:horizontal {
        height: 6px;
        background: #313244;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #89b4fa;
        width: 14px;
        height: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }
    QSlider::sub-page:horizontal {
        background: #89b4fa;
        border-radius: 3px;
    }
"""

SLIDERS = [
    ("Grid Points", "grid_points", 10, 2500, 500, 1),
    ("Decay Iterations", "decay_iters", 1, 16, 4, 1),
    ("Downsample", "downsample", 1, 8, 4, 1),
    ("RBF Smoothing", "rbf_smoothing", 0, 1000, 0, 1),
    ("Point Opacity %", "point_opacity", 0, 100, 50, 1),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.video_label = QLabel()
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        layout.addWidget(self.video_label, stretch=1)

        controls = QVBoxLayout()
        controls.setSpacing(6)

        for name, key, lo, hi, default, step in SLIDERS:
            params[key] = default
            slider, label = self._make_slider(name, key, lo, hi, default, step)
            controls.addLayout(self._slider_row(label, slider))

        layout.addLayout(controls)

    def _make_slider(self, name, key, min_val, max_val, default, step):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setSingleStep(step)
        slider.setValue(default)

        label = QLabel(f"{name}: {default}")
        label.setFixedWidth(160)

        def on_change(v, lbl=label, n=name, k=key):
            lbl.setText(f"{n}: {v}")
            params[k] = v

        slider.valueChanged.connect(on_change)
        return slider, label

    def _slider_row(self, label, slider):
        row = QHBoxLayout()
        row.addWidget(label)
        row.addWidget(slider, stretch=1)
        return row

    def show_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        img = QImage(
            frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(img)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)


_app = None
_window = None


def create():
    global _app, _window
    _app = QApplication(sys.argv)
    _window = MainWindow()
    _window.show()
    return _app, _window


def show_frame(frame):
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _window.show_frame(rgb)


def process_events():
    _app.processEvents()


def is_running():
    return _window.isVisible()
