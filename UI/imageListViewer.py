# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:03:56 2023

@author: szh1213
"""

import os
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (QListWidget, QListWidgetItem, QListView, QWidget,
                             QApplication, QHBoxLayout, QLabel)


class ImageListWidget(QListWidget):
    def __init__(self):
        super(ImageListWidget, self).__init__()
        self.setFlow(QListView.Flow(1))  # 0: left to right,1: top to bottom
        self.setIconSize(QSize(300, 300))

    def add_image_items(self, image_paths=[]):
        for img_path in image_paths:
            if os.path.isfile(img_path):
                img_name = os.path.basename(img_path)
                item = QListWidgetItem(QIcon(img_path), img_name)
                # item.setText(img_name)
                # item.setIcon(QIcon(img_path))
                self.addItem(item)


class ImageViewerWidget(QWidget):
    def __init__(self):
        super(QWidget, self).__init__()
        # 显示控件
        self.list_widget = ImageListWidget()
        self.list_widget.setMinimumWidth(200)
        self.show_label = QLabel(self)
        self.show_label.setFixedSize(600, 400)
        self.image_paths = []
        self.currentImgIdx = 0
        self.currentImg = None

        # 水平布局
        self.layout = QHBoxLayout(self)
        self.layout.addWidget(self.show_label)
        self.layout.addWidget(self.list_widget)

        # 信号与连接
        self.list_widget.itemSelectionChanged.connect(self.loadImage)

    def load_from_paths(self, img_paths=[]):
        self.image_paths = img_paths
        self.list_widget.add_image_items(img_paths)

    def loadImage(self):
        self.currentImgIdx = self.list_widget.currentIndex().row()
        if self.currentImgIdx in range(len(self.image_paths)):
            self.currentImg = QPixmap(
                self.image_paths[self.currentImgIdx]).scaledToHeight(400)
            self.show_label.setPixmap(self.currentImg)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    # 图像路径
    img_dir = r'D:\datahome\OK'
    filenames = os.listdir(img_dir)
    img_paths = []
    for file in filenames:
        if file[-4:] == ".png" or file[-4:] == ".jpg":
            img_paths.append(os.path.join(img_dir, file))

    # 显示控件
    main_widget = ImageViewerWidget()
    main_widget.load_from_paths(img_paths)
    main_widget.setWindowTitle("ImageViewer")
    main_widget.show()

    # 应用程序运行
    sys.exit(app.exec_())
