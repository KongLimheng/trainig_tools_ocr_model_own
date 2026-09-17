"""Interactive Image Viewer Widget with Zoom Toolbar & Baseline Guides."""

from io import BytesIO
from PIL import Image, ImageDraw
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QPushButton, QSlider, QSizePolicy, QCheckBox
)
import qtawesome as qta


class KhmerImageViewer(QWidget):
    """Clean, interactive image viewer with 1:1 actual size, zoom slider, and baseline guidelines."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)

        self.current_pil_image: Image.Image | None = None
        self.current_pixmap: QPixmap | None = None
        self.zoom_factor = 1.0
        self.fit_window_mode = False

        # 1. Top Control Toolbar
        self.toolbar = QWidget()
        self.toolbar.setStyleSheet("background-color: #181825; border: 1px solid #313244; border-radius: 6px; padding: 2px;")
        tb_layout = QHBoxLayout(self.toolbar)
        tb_layout.setContentsMargins(6, 2, 6, 2)
        tb_layout.setSpacing(6)

        self.btn_actual_size = QPushButton("1:1 Actual")
        self.btn_actual_size.setIcon(qta.icon("fa5s.compress", color="#cdd6f4"))
        self.btn_actual_size.setToolTip("Show image at 100% actual pixel resolution")
        self.btn_actual_size.clicked.connect(self._set_actual_size)
        tb_layout.addWidget(self.btn_actual_size)

        self.btn_fit_window = QPushButton("Fit Window")
        self.btn_fit_window.setIcon(qta.icon("fa5s.expand", color="#cdd6f4"))
        self.btn_fit_window.setToolTip("Scale image to fit container window")
        self.btn_fit_window.clicked.connect(self._set_fit_window)
        tb_layout.addWidget(self.btn_fit_window)

        self.btn_zoom_out = QPushButton()
        self.btn_zoom_out.setIcon(qta.icon("fa5s.search-minus", color="#cdd6f4"))
        self.btn_zoom_out.setToolTip("Zoom Out")
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        tb_layout.addWidget(self.btn_zoom_out)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(100)
        self.zoom_slider.valueChanged.connect(self._on_slider_changed)
        tb_layout.addWidget(self.zoom_slider)

        self.btn_zoom_in = QPushButton()
        self.btn_zoom_in.setIcon(qta.icon("fa5s.search-plus", color="#cdd6f4"))
        self.btn_zoom_in.setToolTip("Zoom In")
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        tb_layout.addWidget(self.btn_zoom_in)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #89b4fa; font-weight: bold; min-width: 40px;")
        tb_layout.addWidget(self.zoom_label)

        self.chk_guides = QCheckBox("Guides")
        self.chk_guides.setToolTip("Overlay baseline & ascender/descender guidelines")
        self.chk_guides.stateChanged.connect(self._on_guides_toggled)
        tb_layout.addWidget(self.chk_guides)

        tb_layout.addStretch()
        self.layout.addWidget(self.toolbar)

        # 2. Central Scrollable Canvas
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setStyleSheet(
            "background-color: #11111b; border: 1px solid #313244; border-radius: 6px;"
        )

        self.image_container = QWidget()
        self.container_layout = QVBoxLayout(self.image_container)
        self.container_layout.setAlignment(Qt.AlignCenter)
        self.container_layout.setContentsMargins(10, 10, 10, 10)

        self.image_label = QLabel(self.image_container)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No Image Loaded")
        self.image_label.setStyleSheet("color: #6c7086; font-size: 14px; font-weight: 500;")

        self.container_layout.addWidget(self.image_label)
        self.scroll_area.setWidget(self.image_container)
        self.layout.addWidget(self.scroll_area)

    def set_pil_image(self, pil_image: Image.Image) -> None:
        """Loads and displays a PIL Image."""
        self.current_pil_image = pil_image.copy()
        self._refresh_pixmap()

    def set_image_path(self, path: str) -> None:
        """Loads and displays an image from a file path."""
        try:
            with Image.open(path) as img:
                self.set_pil_image(img)
        except Exception:
            self.current_pixmap = QPixmap(path)
            self.current_pil_image = None
            self._update_display()

    def clear(self) -> None:
        """Clears the displayed image."""
        self.current_pil_image = None
        self.current_pixmap = None
        self.image_label.clear()
        self.image_label.setText("No Image Loaded")

    def _refresh_pixmap(self) -> None:
        if self.current_pil_image is None:
            return

        display_img = self.current_pil_image.copy()

        # Draw baseline & subscript guides if enabled
        if self.chk_guides.isChecked():
            draw = ImageDraw.Draw(display_img)
            w, h = display_img.size
            # Top ascender guide (cyan)
            draw.line([(0, int(h * 0.22)), (w, int(h * 0.22))], fill=(0, 200, 255), width=1)
            # Baseline guide (green)
            draw.line([(0, int(h * 0.70)), (w, int(h * 0.70))], fill=(0, 255, 128), width=1)
            # Subscript bottom limit (yellow)
            draw.line([(0, int(h * 0.92)), (w, int(h * 0.92))], fill=(255, 200, 0), width=1)

        buffer = BytesIO()
        display_img.save(buffer, format="PNG")
        qimage = QImage()
        qimage.loadFromData(buffer.getvalue(), "PNG")
        self.current_pixmap = QPixmap.fromImage(qimage)
        self._update_display()

    def _set_actual_size(self):
        self.fit_window_mode = False
        self.zoom_factor = 1.0
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(100)
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText("100%")
        self._update_display()

    def _set_fit_window(self):
        self.fit_window_mode = True
        self._update_display()

    def _zoom_in(self):
        val = min(400, self.zoom_slider.value() + 25)
        self.zoom_slider.setValue(val)

    def _zoom_out(self):
        val = max(25, self.zoom_slider.value() - 25)
        self.zoom_slider.setValue(val)

    def _on_slider_changed(self, value: int):
        self.fit_window_mode = False
        self.zoom_factor = value / 100.0
        self.zoom_label.setText(f"{value}%")
        self._update_display()

    def _on_guides_toggled(self):
        self._refresh_pixmap()

    def _update_display(self) -> None:
        if self.current_pixmap and not self.current_pixmap.isNull():
            orig_w = self.current_pixmap.width()
            orig_h = self.current_pixmap.height()

            if self.fit_window_mode:
                view_w = max(50, self.scroll_area.viewport().width() - 30)
                view_h = max(50, self.scroll_area.viewport().height() - 30)
                scaled = self.current_pixmap.scaled(
                    view_w, view_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                computed_pct = int((scaled.width() / max(1, orig_w)) * 100)
                self.zoom_label.setText(f"{computed_pct}%")
            else:
                target_w = max(16, int(orig_w * self.zoom_factor))
                target_h = max(16, int(orig_h * self.zoom_factor))
                scaled = self.current_pixmap.scaled(
                    target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )

            self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_window_mode:
            self._update_display()
