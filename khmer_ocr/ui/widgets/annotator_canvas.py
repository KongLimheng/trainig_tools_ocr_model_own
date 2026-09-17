"""Interactive Document Annotation Canvas with Resizable Bounding Boxes."""

from io import BytesIO
from typing import List, Tuple, Optional
from PIL import Image
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QFont, QPainter
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsItem, QLabel, QPushButton, QLineEdit,
    QMessageBox, QSplitter
)
import qtawesome as qta


class ResizableBoxItem(QGraphicsRectItem):
    """An interactive, selectable, and movable/resizable bounding box."""

    def __init__(self, x: float, y: float, w: float, h: float, text: str = "", index: int = 1):
        super().__init__(x, y, w, h)
        self.text = text
        self.index = index

        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._normal_pen = QPen(QColor(59, 130, 246, 220), 2)
        self._selected_pen = QPen(QColor(245, 158, 11, 255), 2.5)
        self._normal_brush = QBrush(QColor(59, 130, 246, 30))
        self._selected_brush = QBrush(QColor(245, 158, 11, 50))

        self.setPen(self._normal_pen)
        self.setBrush(self._normal_brush)

    def paint(self, painter, option, widget=None):
        if self.isSelected():
            self.setPen(self._selected_pen)
            self.setBrush(self._selected_brush)
        else:
            self.setPen(self._normal_pen)
            self.setBrush(self._normal_brush)

        super().paint(painter, option, widget)

        # Draw line index badge
        painter.save()
        rect = self.rect()
        badge_rect = QRectF(rect.x(), rect.y() - 18, 24, 18)
        painter.fillRect(badge_rect, QColor(59, 130, 246, 220))
        painter.setPen(Qt.white)
        painter.setFont(QFont("sans-serif", 9, QFont.Bold))
        painter.drawText(badge_rect, Qt.AlignCenter, str(self.index))
        painter.restore()


class DocumentAnnotatorWidget(QWidget):
    """Interactive Document Annotation Studio with visual bounding boxes & label editing."""

    box_selected = pyqtSignal(int, str, QRectF)  # (index, text, rect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image: Image.Image | None = None
        self.box_items: List[ResizableBoxItem] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Top Action Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background-color: #181825; border: 1px solid #313244; border-radius: 6px; padding: 2px;")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(6, 4, 6, 4)

        self.btn_auto_detect = QPushButton("Auto-Detect Lines")
        self.btn_auto_detect.setIcon(qta.icon("fa5s.magic", color="#cdd6f4"))
        self.btn_auto_detect.clicked.connect(self._auto_detect)
        tb_layout.addWidget(self.btn_auto_detect)

        self.btn_add_box = QPushButton("Add Box")
        self.btn_add_box.setIcon(qta.icon("fa5s.plus", color="#cdd6f4"))
        self.btn_add_box.clicked.connect(self._add_box)
        tb_layout.addWidget(self.btn_add_box)

        self.btn_delete_box = QPushButton("Delete Box")
        self.btn_delete_box.setIcon(qta.icon("fa5s.trash", color="#f38ba8"))
        self.btn_delete_box.clicked.connect(self._delete_selected_box)
        tb_layout.addWidget(self.btn_delete_box)

        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.setIcon(qta.icon("fa5s.times", color="#cdd6f4"))
        self.btn_clear_all.clicked.connect(self._clear_boxes)
        tb_layout.addWidget(self.btn_clear_all)

        tb_layout.addStretch()
        self.count_label = QLabel("0 boxes")
        self.count_label.setStyleSheet("color: #89b4fa; font-weight: bold;")
        tb_layout.addWidget(self.count_label)

        layout.addWidget(toolbar)

        # Splitter: Canvas & Edit Box
        splitter = QSplitter(Qt.Horizontal)

        # Graphics Scene & View
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setStyleSheet("background-color: #11111b; border: 1px solid #313244; border-radius: 6px;")
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.scene.selectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.view)

        # Side Panel: Active Box Label Editor
        side_panel = QWidget()
        side_panel.setFixedWidth(280)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(4, 4, 4, 4)

        side_layout.addWidget(QLabel("Selected Line Label:"))
        self.edit_text = QLineEdit()
        self.edit_text.setStyleSheet("font-size: 14px; padding: 6px;")
        self.edit_text.textChanged.connect(self._on_text_edited)
        side_layout.addWidget(self.edit_text)

        self.info_label = QLabel("Click any box on the canvas to select, move, or resize.")
        self.info_label.setStyleSheet("color: #a6adc8; font-size: 11.5px;")
        self.info_label.setWordWrap(True)
        side_layout.addWidget(self.info_label)

        side_layout.addStretch()
        splitter.addWidget(side_panel)
        splitter.setSizes([720, 280])

        layout.addWidget(splitter)

    def load_document_image(self, image: Image.Image, auto_detect: bool = True):
        """Loads a document image onto the canvas."""
        self.current_image = image.copy()
        self.scene.clear()
        self.box_items.clear()

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        qimage = QImage()
        qimage.loadFromData(buffer.getvalue(), "PNG")
        pixmap = QPixmap.fromImage(qimage)

        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

        if auto_detect:
            self._auto_detect()

    def _auto_detect(self):
        if self.current_image is None:
            return

        from ...pipeline.segmenter import KhmerDocumentLineSegmenter
        segmenter = KhmerDocumentLineSegmenter()
        bboxes, _, _ = segmenter.segment(self.current_image)

        self._clear_boxes()
        for idx, (x, y, w, h) in enumerate(bboxes):
            box = ResizableBoxItem(x, y, w, h, index=idx + 1)
            self.scene.addItem(box)
            self.box_items.append(box)

        self.count_label.setText(f"{len(self.box_items)} boxes")

    def _add_box(self):
        if self.current_image is None:
            return
        idx = len(self.box_items) + 1
        box = ResizableBoxItem(50, 50 + (idx * 30) % 300, 250, 48, index=idx)
        self.scene.addItem(box)
        self.box_items.append(box)
        box.setSelected(True)
        self.count_label.setText(f"{len(self.box_items)} boxes")

    def _delete_selected_box(self):
        selected = self.scene.selectedItems()
        for item in selected:
            if isinstance(item, ResizableBoxItem):
                self.scene.removeItem(item)
                if item in self.box_items:
                    self.box_items.remove(item)

        # Re-index remaining boxes
        for i, b in enumerate(self.box_items):
            b.index = i + 1
            b.update()

        self.count_label.setText(f"{len(self.box_items)} boxes")
        self.edit_text.clear()

    def _clear_boxes(self):
        for item in self.box_items:
            self.scene.removeItem(item)
        self.box_items.clear()
        self.count_label.setText("0 boxes")
        self.edit_text.clear()

    def _on_selection_changed(self):
        selected = self.scene.selectedItems()
        if selected and isinstance(selected[0], ResizableBoxItem):
            box = selected[0]
            self.edit_text.blockSignals(True)
            self.edit_text.setText(box.text)
            self.edit_text.blockSignals(False)
            self.info_label.setText(
                f"Box #{box.index} selected\nX: {int(box.x())}, Y: {int(box.y())}\nW: {int(box.rect().width())}, H: {int(box.rect().height())}"
            )
            self.box_selected.emit(box.index, box.text, box.sceneBoundingRect())

    def _on_text_edited(self, new_text: str):
        selected = self.scene.selectedItems()
        if selected and isinstance(selected[0], ResizableBoxItem):
            selected[0].text = new_text

    def get_annotated_records(self) -> List[Tuple[Image.Image, str]]:
        """Returns list of (cropped_line_pil_image, text_label)."""
        if self.current_image is None:
            return []

        results = []
        # Sort boxes top to bottom
        sorted_boxes = sorted(self.box_items, key=lambda b: b.sceneBoundingRect().y())

        for b in sorted_boxes:
            rect = b.sceneBoundingRect()
            x1 = max(0, int(rect.x()))
            y1 = max(0, int(rect.y()))
            x2 = min(self.current_image.width, int(rect.x() + rect.width()))
            y2 = min(self.current_image.height, int(rect.y() + rect.height()))

            if x2 > x1 and y2 > y1:
                crop = self.current_image.crop((x1, y1, x2, y2))
                results.append((crop, b.text.strip()))

        return results
