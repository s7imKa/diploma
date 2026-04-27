from typing import Optional
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem


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
        self.reset_zoom()

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
        zoom_in_factor = 1.2
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self._scale *= zoom_factor
        self.scale(zoom_factor, zoom_factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep image visible when the widget is resized
        if self._pixmap_item is not None and self._image is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._scale = 1.0
