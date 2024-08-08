# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:38:20 2023

@author: szh1213
"""

from PyQt5.QtCore import (pyqtSignal, QPoint, QPointF, Qt, QRectF)
from PyQt5.QtWidgets import (QGraphicsPixmapItem, QGraphicsItem,
                             QGraphicsScene, QGraphicsView, QFrame, )
from PyQt5.QtGui import (QPixmap, QFont, QColor, QBrush, QPen)


class PhotoViewer(QGraphicsView):
    photoClicked = pyqtSignal(QPoint)
    itemClicked = pyqtSignal(tuple)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.picInfo = self._scene.addText('', QFont('KaiTi', 40))
        self.picInfo.setPos(50, 50)
        self.picInfo.setDefaultTextColor(QColor(0, 255, 0))
        self.boxes = []
        self.boxNames = []
        self.boxInfo = []

        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)
        self.setVisible(False)
        self.setMinimumWidth(300)
        self.viewport().setMinimumWidth(300)
        self.viewport().setMinimumHeight(300)

    def hasPhoto(self):  # 检查是否有图片加载
        return not self._empty

    def fitInView(self, scale=True):  # 调整视图以适应图片大小
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()

                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
            self.setVisible(True)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        self.fitInView()

    def setInfo(self, text):
        self.picInfo.setPlainText(text)

    def addBox(self, name, x, y, w, h, qcolor):
        pen = QPen(qcolor)
        t = self._scene.addRect(x, y, w, h, pen, qcolor)
        if name in ["wbc", "po", "pf", "pv", "pv-s", "pm", 'malaria']:
            info_ = self._scene.addText(name.upper(), QFont('KaiTi', 40))
            info_.setPos(x, y)
            self.boxInfo.append(info_)

        self.boxes.append(t)
        self.boxNames.append(name)

    def clearBoxes(self):
        for box in self.boxes:
            self._scene.removeItem(box)
        self.boxes.clear()
        for box_ in self.boxInfo:
            self._scene.removeItem(box_)

    def getItemAtClick(self, pos):  # getItemAtClick 方法获取在特定位置点击的图元
        """ 获取点击位置的图元，无则返回None. """

        item = self.itemAt(pos)
        if isinstance(item, QGraphicsItem) and item in self.boxes:
            index = self.boxes.index(item)
            return (index, self.boxNames[index])
        return (-1, '')

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):  # mousePressEvent 方法处理鼠标点击事件，并发出相应的信号
        try:
            t = self.mapToScene(event.position().toPoint())
        except:
            t = self.mapToScene(event.pos())
            if type(t) == QPointF:
                t = t.toPoint()
        if self._photo.isUnderMouse():
            self.photoClicked.emit(t)
        index, name = self.getItemAtClick(event.pos())
        self.itemClicked.emit((index, name))
        super(PhotoViewer, self).mousePressEvent(event)
