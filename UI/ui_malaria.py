# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'malaria.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(926, 729)
        self.widget = QtWidgets.QWidget(MainWindow)
        self.widget.setObjectName("widget")
        self.topHL = QtWidgets.QHBoxLayout(self.widget)
        self.topHL.setObjectName("topHL")
        MainWindow.setCentralWidget(self.widget)
        self.dockWidget_2 = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget_2.setObjectName("dockWidget_2")
        self.dockWidgetContents_2 = QtWidgets.QWidget()
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.dockWidgetContents_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.openPicDirBTN = QtWidgets.QPushButton(self.dockWidgetContents_2)
        self.openPicDirBTN.setLayoutDirection(QtCore.Qt.RightToLeft)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("foreland.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.openPicDirBTN.setIcon(icon)
        self.openPicDirBTN.setObjectName("openPicDirBTN")
        self.verticalLayout_2.addWidget(self.openPicDirBTN)
        self.foregroundBTN = QtWidgets.QPushButton(self.dockWidgetContents_2)
        self.foregroundBTN.setObjectName("foregroundBTN")
        self.verticalLayout_2.addWidget(self.foregroundBTN)
        self.pushButton_3 = QtWidgets.QPushButton(self.dockWidgetContents_2)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_2.addWidget(self.pushButton_3)
        self.dockWidget_2.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_2)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.openPicDirBTN.setText(_translate("MainWindow", "打开文件夹"))
        self.foregroundBTN.setText(_translate("MainWindow", "前景"))
        self.pushButton_3.setText(_translate("MainWindow", "PushButton"))
