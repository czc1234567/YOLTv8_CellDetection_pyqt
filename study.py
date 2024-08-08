from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import time

class WorkerThread(QThread):
    finished = pyqtSignal()

    def run(self):
        # 模拟耗时任务
        time.sleep(5)  # 假设任务需要5秒钟
        self.finished.emit()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.startButton = QPushButton('开始检测', self)
        self.lockButton = QPushButton('锁定按钮', self)

        self.startButton.clicked.connect(self.on_start_detection)
        self.lockButton.clicked.connect(self.on_lock_button)

        layout = QVBoxLayout()
        layout.addWidget(self.startButton)
        layout.addWidget(self.lockButton)
        self.setLayout(layout)
        self.setWindowTitle('按钮控制示例')
        self.show()

    def on_start_detection(self):
        # 开始检测时，禁用锁定按钮
        self.lockButton.setEnabled(False)

        # 创建并启动工作线程
        self.thread = WorkerThread()
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()

    def on_thread_finished(self):
        # 任务完成后，重新启用锁定按钮
        self.lockButton.setEnabled(True)

    def on_lock_button(self):
        # 锁定按钮被点击时，禁用开始检测按钮
        self.startButton.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())