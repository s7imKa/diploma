from typing import Optional
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
)


class FullscreenImageDialog(QDialog):
    """Full-screen image viewer dialog opened on double-click."""

    def __init__(self, img: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Перегляд зображення — подвійний клік або Esc для закриття")
        self.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        hint = QLabel("🔍 Колесо миші — зум   |   ЛКМ — переміщення   |   Esc / F11 — закрити")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #aaa; font-size: 9pt; padding: 2px;")
        layout.addWidget(hint)

        self._viewer = ImageViewer(self)
        self._viewer.set_image(img)
        layout.addWidget(self._viewer, 1)

        btn_close = QPushButton("✕ Закрити")
        btn_close.setFixedHeight(28)
        btn_close.clicked.connect(self.close)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self.setLayout(layout)
        self.showMaximized()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_F11):
            self.close()
        super().keyPressEvent(event)


class ImageViewer(QGraphicsView):
    """Simple zoomable/pannable viewer for numpy images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._image: Optional[np.ndarray] = None
        self._scale = 1.0
        self.setToolTip("⌃ Ctrl + прокрутка — зум  |  Два пальці — панорамування  |  Подвійний клік — повний екран")

    def set_image(self, img: np.ndarray):
        if img is None:
            self.clear()
            return
        self._image = img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(0, 0, w, h)
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._scale = 1.0

    def clear(self):
        self._scene.clear()
        self._pixmap_item = None
        self._image = None
        self._scale = 1.0

    def reset_zoom(self):
        self.resetTransform()
        self._scale = 1.0

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return
        if event.modifiers() & Qt.ControlModifier:
            zoom_in_factor = 1.2
            zoom_out_factor = 1 / zoom_in_factor
            zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
            self._scale *= zoom_factor
            self.scale(zoom_factor, zoom_factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep image visible when the widget is resized
        if self._pixmap_item is not None and self._image is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._scale = 1.0

    def mouseDoubleClickEvent(self, event):
        if self._image is not None:
            dialog = FullscreenImageDialog(self._image, self.window())
            dialog.exec_()
        super().mouseDoubleClickEvent(event)
