"""
PyQt6 Interface

Window with video display and parameter sliders.
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

# === PARAMETERS ===
WINDOW_TITLE = "Smooth Webcam"
WINDOW_WIDTH = 860
WINDOW_HEIGHT = 580

MAX_FEATURES_DEFAULT = 500
MAX_FEATURES_MIN = 10
MAX_FEATURES_MAX = 1000

FILTER_RC_DEFAULT = 500
FILTER_RC_MIN = 0
FILTER_RC_MAX = 2000

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

        self.features_slider, self.features_label = self._make_slider(
            "Max Features",
            MAX_FEATURES_MIN,
            MAX_FEATURES_MAX,
            MAX_FEATURES_DEFAULT,
        )
        controls.addLayout(self._slider_row(self.features_label, self.features_slider))

        self.rc_slider, self.rc_label = self._make_slider(
            "Filter RC (ms)",
            FILTER_RC_MIN,
            FILTER_RC_MAX,
            FILTER_RC_DEFAULT,
        )
        controls.addLayout(self._slider_row(self.rc_label, self.rc_slider))

        layout.addLayout(controls)

    def _make_slider(
        self,
        name,
        min_val,
        max_val,
        default,
    ):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)

        label = QLabel(f"{name}: {default}")
        label.setFixedWidth(160)

        slider.valueChanged.connect(
            lambda v, lbl=label, n=name: lbl.setText(f"{n}: {v}")
        )

        return slider, label

    def _slider_row(
        self,
        label,
        slider,
    ):
        row = QHBoxLayout()
        row.addWidget(label)
        row.addWidget(slider, stretch=1)
        return row

    def show_frame(
        self,
        frame,
    ):
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

    def read_params(self):
        return (
            self.features_slider.value(),
            self.rc_slider.value(),
        )


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


def read_params():
    return _window.read_params()


def process_events():
    _app.processEvents()


def is_running():
    return _window.isVisible()
