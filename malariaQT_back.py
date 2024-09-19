# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:23:34 2023.

@author: szh1213
"""
import copy
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
import os
import json
import sys
import pickle
from PyQt5.QtGui import (QIcon, QStandardItemModel, QImage, QStandardItem,
                         QPixmap, QColor, QFont)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint, QSettings
from PyQt5.QtWidgets import (QVBoxLayout, QMainWindow, QTableView, QWidget, QComboBox, QDialog,
                             QAbstractItemView, QApplication, QPushButton, QVBoxLayout, QSlider, QShortcut,
                             QFileDialog, QDockWidget, QTextEdit, QListView,
                             QProgressBar, QAction, QLabel, QWidgetAction,
                             QHBoxLayout, QHeaderView, QMessageBox)
from UI.PhotoViewer import PhotoViewer
from core.myAlgo import findCell, cvimread, cvimwrite, infoDict
from core.inferenceApi import inferenceC as loadClassifier
from core.inferenceApi import inferenceD as loadDetecter
from core.inferenceApi import inferenceResnet as loadClassifierResnet
from core.prefunc import prefunc, compute_IOU
from malaria_inference import malaria_inference
import ctypes
import ctypes.wintypes
import torch
import time
import filetype
from user import checkKeyCode

_ = os.path.split(__file__)[1]
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_)
start = time.time()
DEVICE = 0 if torch.cuda.device_count() else 'cpu'

print('device=', DEVICE)
import os

os.environ['PYTHONTHREADSTACKSIZE'] = '104857600'


class ConfidenceDialog(QDialog):
    # 定义信号，用于通知外部置信度值改变
    confidenceChanged = pyqtSignal(int)
    # 定义信号，用于通知外部置信度值被提交（例如，用户关闭对话框）
    valueCommitted = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        # 设置应用程序的组织名称和应用程序名称，用于QSettings
        self.settings = QSettings("YourCompany", "YourApp")
        # 默认值
        self.Value = 65
        # 从QSettings加载置信度值，如果不存在则使用默认值
        self.currentValue = self.settings.value("confidenceLevel", self.Value)
        print(self.currentValue)  # 打印当前加载的置信度值
        self.initUI()  # 初始化用户界面

    def initUI(self):
        # 设置对话框标题和窗口标志
        self.setWindowTitle('Confidence Level Adjuster')
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        # 设置对话框样式
        self.setStyleSheet("""
            QDialog {
                background-color: #282c34;
                color: #ffffff;
            }
            QSlider {
                border: 1px solid #ffffff;
                background: #3a3f4b;
                height: 20px;
            }
            QSlider::handle:horizontal {
                background: #61afef;
                border: 1px solid #5a95e3;
                width: 20px;
                margin: -8px 0;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #3a3f4b;
                border: 1px solid #5a95e3;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        # 创建水平滑动条，设置最小值、最大值和初始值
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(self.currentValue)
        # 连接滑动条的值改变信号到更新当前值和标签的槽函数，以及发射置信度改变信号
        self.slider.valueChanged.connect(self.updateCurrentValue)
        self.slider.valueChanged.connect(self.updateLabel)
        self.slider.valueChanged.connect(self.confidenceChanged.emit)

        # 创建标签显示当前置信度值
        self.label = QLabel('Confidence Level:{}'.format(self.currentValue))
        self.label.setAlignment(Qt.AlignCenter)

        # 创建布局并添加滑动条和标签到对话框
        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def updateCurrentValue(self, value):
        # 更新当前置信度值
        self.currentValue = value

    def updateLabel(self, value):
        # 更新标签显示的置信度值
        self.label.setText(f'Confidence Level: {value}%')

    def accept(self):
        # 当用户接受对话框时，保存当前值到QSettings并发射提交信号
        self.settings.setValue("confidenceLevel", self.slider.value())
        self.valueCommitted.emit(self.slider.value())
        super().accept()

    def reject(self):
        # 当用户拒绝对话框时，保存当前值到QSettings并发射提交信号
        self.settings.setValue("confidenceLevel", self.slider.value())
        self.valueCommitted.emit(self.slider.value())
        super().reject()

    def showEvent(self, event):
        # 当对话框显示时，从QSettings重新加载置信度值并更新滑动条
        super().showEvent(event)
        self.currentValue = self.settings.value("confidenceLevel", self.Value)
        self.slider.setValue(self.currentValue)

    def getValue(self):
        # 返回当前滑动条的值
        return self.slider.value()


class malariaThread(QThread):
    trigger = pyqtSignal(dict)
    auto = pyqtSignal(int)
    results = pyqtSignal(object)

    loop = False

    def __init__(self, current_country):
        super().__init__()
        # model
        self.thinDetecter = None
        self.thickDetecter = None
        self.input_image = None
        self.batch_path = ''
        self.thinkness = ''
        # 原始版CellDetect
        self.malariaCellDetecter = loadDetecter('weights/CellDetect0619.pt', 0.1, 0.45)
        self.malariathicknessClassifier = loadClassifierResnet('weights/Classifier.pth')
        """初始化，计数."""
        self.pic = -1
        self.thickness = 'thin'
        self.mode = 2
        # 初始化国家信息
        self.current_country = current_country
        self.conf = 0.25

    #主线程
    def run(self) -> None:

        print("多线程run")
        print(time.time())


        # 初始化 参数
        PF, PM, PO, POPV, PV = 0, 0, 0, 0, 0
        conf_numbers = 0
        file_numbers = 0
        patient_status = '阴性'
        # 加载模型
        model = YOLO("weights/classify_9.9.pt")

        crop_path = Path('dataset/thin/crop')
        if crop_path.exists() and crop_path.is_dir():
            # 遍历目录下的所有文件和子目录
            for file_path in crop_path.iterdir():
                # 如果是文件，则删除该文件
                if file_path.is_file():
                    file_path.unlink()
                # 如果是目录，则删除该目录及其所有内容
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
        else:
            # 如果路径不存在或不是一个目录，则打印提示信息
            print(f"路径 {crop_path} 不存在或不是一个目录。")

        # 继续执行后续代码
        path = crop_path

        # 检查路径下是否有文件 防止模型训练时没有数据
        def check_path_not_empty(path):
            if not os.path.exists(path):
                print(f"路径 {path} 不存在。")
                return False
            if not os.listdir(path):
                print(f"路径 {path} 为空。")
                return False
            return True




        batch_info = {}
        file_index = -1
        results_info = []

        save_folder = "output"
        if os.path.exists(save_folder):
            # 遍历文件夹中的所有内容并删除
            for filename in os.listdir(save_folder):
                file_path = os.path.join(save_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            # 创建文件夹
            os.makedirs(save_folder, exist_ok=True)
        print(f"Folder '{save_folder}' is ready to use.")

        files = os.listdir(self.batch_path)
        sorted_files = sorted(files, key=lambda x: os.path.join(self.batch_path, x))
        for file in sorted_files:
            # for file in os.listdir(self.batch_path):
            cell_info_dict = {}
            px_info_dict = {}
            file_index += 1
            self.pic = file_index

            key = 0
            imgpath = os.path.join(self.batch_path, file)
            matbgr = cvimread(imgpath)
            matyolo = prefunc(matbgr)
            thinkness_res = self.malariathicknessClassifier.infer(matbgr)


            # 判断是薄片还是厚片
            if self.thinkness == "thin":
                cell_pred = self.malariaCellDetecter.infer(matyolo, device=DEVICE)
                # 检测细胞
                for xywh, class_, conf in zip(cell_pred.boxes.xywh,
                                              cell_pred.boxes.cls,
                                              cell_pred.boxes.conf):
                    # self.cellInfos[self.picSeed][key] = {}
                    cell_info_dict[key] = {}
                    cx, cy, w, h = map(int, xywh)

                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    # if "cell" not in self.infos[self.picSeed]["细胞"]:
                    #     self.infos[self.picSeed]["细胞"]["cell"] = 0
                    # else:
                    #     self.infos[self.picSeed]["细胞"]["cell"] += 1
                    # self.cellInfos[self.picSeed][key]['类型'] = "cell"
                    # self.cellInfos[self.picSeed][key]['置信度'] = conf.item()
                    # self.cellInfos[self.picSeed][key]['位置矩形'] = (x, y, w, h)
                    cell_info_dict[key]['类型'] = "cell"
                    cell_info_dict[key]['置信度'] = conf.item()
                    cell_info_dict[key]['位置矩形'] = (x, y, w, h)
                    key += 1
                # 检测疟原虫

            if self.thinkness == "thin":
                curent_model = self.thinDetecter
                px_pred, conf_number, file_number = curent_model.infer(imgpath)

                conf_numbers += conf_number
                file_numbers += file_number

                for xywh, class_, conf in zip(px_pred.boxes.xywh,
                                              px_pred.boxes.cls,
                                              px_pred.boxes.conf):
                    # self.cellInfos[self.picSeed][key] = {}
                    cx, cy, w, h = map(int, xywh)
                    if int(class_) == 0:
                        px_info_dict[key] = {}
                        class_wbc = px_pred.names[int(class_)].lower()
                        x = int(cx - w / 2)
                        y = int(cy - h / 2)
                        px_info_dict[key]['类型'] = class_wbc
                        px_info_dict[key]['置信度'] = conf.item()
                        px_info_dict[key]['位置矩形'] = (x, y, w, h)
                        key += 1
                        continue
                    if w < 2 * h and h < 2 * w and w * h < 150000 and w * h > 25600:
                        px_info_dict[key] = {}
                        class_ = px_pred.names[int(class_)].lower()

                        x = int(cx - w / 2)
                        y = int(cy - h / 2)
                        # self.cellInfos[self.picSeed][key]['类型'] = "cell"
                        # self.cellInfos[self.picSeed][key]['置信度'] = conf.item()
                        # self.cellInfos[self.picSeed][key]['位置矩形'] = (x, y, w, h)

                        # 此处为虫型细分
                        # px_info_dict[key]['类型'] = class_

                        if class_ in ['pv', 'pm', 'po', 'pf']:
                            class_ = 'malaria'

                        # 此处强行将细分  改为  疟原虫
                        px_info_dict[key]['类型'] = class_
                        px_info_dict[key]['置信度'] = conf.item()
                        px_info_dict[key]['位置矩形'] = (x, y, w, h)
                        key += 1


                batch_info[file_index] = {"imgseed": file_index, "thinkness": thinkness_res[0],
                                          "thinkness_conf": thinkness_res[1], "info": {}}

                # if self.picSeed not in self.detected_seed:
                #     self.detected_seed.append(self.picSeed)

                # print(cell_info_dict)
                # print(px_info_dict)

                # 对cell和Px分别处理
                max_index = 0
                for cckey, ccvalue in cell_info_dict.items():
                    batch_info[file_index]["info"][cckey] = ccvalue
                    max_index = cckey + 1
                    for ppkey, ppvalue in px_info_dict.items():
                        if compute_IOU(ccvalue["位置矩形"], ppvalue["位置矩形"]) >= 0.5:
                            # batch_info[file_index]["info"][cckey] = ppvalue

                            batch_info[file_index]["info"][cckey] = ppvalue

                            del px_info_dict[ppkey]
                            break
                for pppkey, pppvalue in px_info_dict.items():
                    batch_info[file_index]["info"][max_index] = pppvalue
                """逐个检测展示细胞."""
                self.trigger.emit(batch_info[file_index])


            else:
                result_malaria = self.thickDetecter.infer(imgpath)
                for xywh, class_, conf in zip(result_malaria.boxes.xywh,
                                              result_malaria.boxes.cls,
                                              result_malaria.boxes.conf):
                    # self.cellInfos[self.picSeed][key] = {}
                    # print(class_)
                    px_info_dict[key] = {}

                    class_ = result_malaria.names[int(class_)].lower()
                    cx, cy, w, h = map(int, xywh)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    if class_ in ['pv', 'pm', 'po', 'pf']:
                        class_ = 'malaria'

                    # 此处强行将细分  改为  疟原虫
                    px_info_dict[key]['类型'] = class_
                    px_info_dict[key]['置信度'] = conf.item()
                    px_info_dict[key]['位置矩形'] = (x, y, w, h)
                    key += 1


                batch_info[file_index] = {"imgseed": file_index, "thinkness": thinkness_res[0],
                                          "thinkness_conf": thinkness_res[1], "info": {}}
                for ppkey, ppvalue in px_info_dict.items():
                    batch_info[file_index]["info"][ppkey] = ppvalue


                self.trigger.emit(batch_info[file_index])




        if self.thinkness == "thin":
            PF_percent = int(0)  # 假设这是某个百分比
            PM_percent = int(0)  # 假设这是某个百分比
            PO_percent = int(0)  # 假设这是某个百分比
            PV_percent = int(0)  # 假设这是某个百分比
            max_key = ''
            if check_path_not_empty(path):
                results = model(path)
                for result in results:
                    probs = result.probs.top1
                    name = results[0].names[int(probs)]

                    n = 0
                    for top in result.probs.top5:
                        result_name = result.names[int(top)]
                        conf = result.probs.top5conf[n]
                        n = n + 1
                        print(f"{result_name} 的置信度为 {conf:.2f}")
                        cv2.putText(result.orig_img, f"{result_name} {conf:.2f}", (10, 20 * n), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0), 1)
                        original_file_name = os.path.basename(result.path)
                        save_path = os.path.join(save_folder, original_file_name)
                        cv2.imwrite(save_path, result.orig_img)

                    if name == 'POPV':
                        POPV += 1
                    elif name == 'PF':
                        PF += 1
                    elif name == 'PM':
                        PM += 1
                    elif name == 'PO':
                        PO += 1
                    elif name == 'PV':
                        PV += 1

                Total_malaria = PF + PM + PO + PV + POPV
                percentages = {
                    # 'PF': (PF / Total_malaria) * 100,
                    'PM': ((PM) / Total_malaria) * 100,
                    'PO': ((PO) / Total_malaria) * 100,
                    'PV': ((PV) / Total_malaria) * 100,
                    'POPV': ((POPV) / Total_malaria) * 100
                }
                PF_percent = int(PF / Total_malaria * 100)
                PM_percent = int(PM / Total_malaria * 100)
                PO_percent = int(PO / Total_malaria * 100)
                PV_percent = int(PV / Total_malaria * 100)
                POPV_percent = int(POPV / Total_malaria * 100)

                # 处理POPV
                if PO_percent >= PV_percent:
                    PO_percent = PO_percent + POPV_percent
                else:
                    PV_percent = PV_percent + POPV_percent

                # max_key = max(percentages, key=percentages.get)

                if PF_percent >= 80:
                    max_key = 'PF'
                else:
                    max_key = max(percentages, key=percentages.get)
                    if max_key == 'POPV':
                        max_key = max(['PO', 'PV'], key=percentages.get)


                if file_numbers >= int(file_index/20) or conf_numbers >= 1:
                    patient_status = '阳性'

                print(f"当前病人为：{patient_status}")
                for k, v in percentages.items():
                    print(f"{k}的概率是 {v:.2f}")
                print('------------------------')
                print(f"判断当前虫型为 {max_key}")
                print(Total_malaria)
            else:
                print(f"当前病人为：{patient_status}")

            # PF = (PF / Total_malaria) * 100
            if patient_status == '阳性':
                results_info = [patient_status, max_key, PF_percent, PM_percent, PO_percent, PV_percent]
            else:
                max_key = ''
                results_info = [patient_status, max_key, 0, 0, 0, 0]
            self.results.emit(results_info)

            # 睡眠两秒是为了让进度条先执行完毕
            self.sleep(2)

            print("多线程end")




    def run_once(self, img_or_path, thinkness):
        # if type(img_or_path) == str:
        #     img = cv2.imread(img_or_path)
        # else:
        #     img = img_or_path
        conf_number, file_number = 0, 0
        print(self.conf)
        if thinkness == "thin":
            pred, conf_number, file_number = self.thinDetecter.infer(img_or_path)
        else:
            pred = self.thickDetecter.infer(img_or_path)
        return pred, conf_number, file_number

    def load_weight(self, thin_weight, thick_weight):
        self.thinDetecter = malaria_inference(thin_weight, self.conf, 0.45)
        self.thickDetecter = loadDetecter(thick_weight, 0.5, 0.45)
        self.thin_weight_path = thin_weight
        self.thick_weight_path = thick_weight

    def load_thin_weight(self, path):
        # self.thinDetecter = loadDetecter(path, 0.5, 0.45)
        self.thinDetecter = malaria_inference(path, self.conf, 0.45)
        print("change to thin model weight to:", path)
        self.thin_weight_path = path

    def load_thick_weight(self, path):
        self.thickDetecter = loadDetecter(path, 0.7, 0.45)
        print("change to thick model weight to:", path)
        self.thick_weight_path = path


class MainWindow(QMainWindow):
    """主窗口类."""

    def __init__(self):
        # move()方法移动了窗口到屏幕坐标x=300, y=300的位置.
        super(MainWindow, self).__init__()

        # 机器码
        machine_code = "7BA4E7D187C65A2EBE993C8257743487"
        # 授权码
        key_code = "MCDJE3GV5C23LWNNO7WQYEOZ5SW4CKSG5M6JOJVXJNL5CRBNEWMOQKR3FIFVGTQVIDB7WWHLK47DBGV5FPMKKDNA2GHD6HETRTBFF6W23SHRQY4O26LLV7I7BCTH5PAR"
        # 授权校验
        checkKeyCode(machine_code, key_code)

        self.process = {}
        self.PID = {}
        self.picDir = None
        self.pics = []
        self.picSeed = -1
        self.picNum = 0
        self.initInfo()
        self.curimg = QPixmap()
        self.actions = []
        self.data = []
        self.mat = {
            'bgr': None,
        }
        self.current_country = "选择国家"  # 当前选中的国家名称
        # self.qcolors = {
        #     'Parasitized'.lower(): QColor(114, 233, 81, 150),
        #     'parasites': QColor(233, 0, 0, 120),
        #     'Uninfected'.lower(): QColor(0, 0, 0, 50),
        #     'Unifected'.lower(): QColor(0, 0, 0, 50),
        #     'cell'.lower(): QColor(0, 0, 0, 50),
        #     'White_Blood_Cell'.lower(): QColor(255, 255, 255, 180),
        #     'wbc': QColor(255, 255, 255, 180),
        #     'pv': QColor(233, 0, 0, 120),
        #     'Plasmodium'.lower(): QColor(233, 0, 0, 120),
        # }
        self.qcolors = {
            'wbc'.lower(): QColor(255, 255, 255, 200),
            'po': QColor(255, 0, 0, 64),
            'malaria'.lower(): QColor(0, 122, 255, 64),
            'pf'.lower(): QColor(0, 0, 255, 64),
            'pv'.lower(): QColor(255, 0, 255, 64),
            'pv-s'.lower(): QColor(0, 255, 255, 64),
            'pm'.lower(): QColor(255, 255, 0, 64),
            'cell': QColor(0, 255, 0, 32),
            'Parasitized'.lower(): QColor(114, 233, 81, 150),
            'parasites': QColor(233, 0, 0, 120),
            'Uninfected'.lower(): QColor(0, 0, 0, 50),
            'Unifected'.lower(): QColor(0, 0, 0, 50),
            'White_Blood_Cell'.lower(): QColor(255, 255, 255, 180),
            'Plasmodium'.lower(): QColor(233, 0, 0, 120),
        }

        # self.cellClasses = ["wbc", "po", "pf", "pv", "pv-s", "pm", 'malaria']

        self.cellClasses = ["wbc", 'malaria']
        self.current_key = 0
        self.detected_seed = []

        # new 加载模型线程
        self.malariaThread = malariaThread(self.current_country)

        # 加载隐藏界面和置信度初始化
        self.confidenceDialog = ConfidenceDialog(self)
        self.malariaThread.conf = self.confidenceDialog.currentValue / 100
        print('----')
        print(self.malariaThread.conf)

        self.malariaThread.load_weight("weights/thin_9.4_detection.pt", "weights/thick-00.pt")
        self.malariaThread.trigger.connect(self.showOneCell)
        self.malariaThread.auto.connect(self.detectCells)
        self.malariaThread.results.connect(self.showResults)

    def loadWeight(self):
        """加载模型."""
        self.thicknessClassifier = loadClassifierResnet('weights/Classifier.pth')
        self.cellDetecter = loadDetecter('weights/CellDetect.pt', 0.1, 0.45)
        self.thinWeights["thin_9.4_detection.pt"].setChecked(True)
        self.thickWeights["thick-00.pt"].setChecked(True)

    def initInfo(self):
        """初始化信息字典."""
        self.infos = {}
        self.cellInfos = {}
        self.peopleInfo = {}

    def initUI(self):
        """初始化."""
        self.takeCentralWidget()
        # 启动dock 嵌套
        self.setDockNestingEnabled(True)
        # 允许设置成Tab窗口样式
        self.setDockOptions(self.dockOptions() |
                            QMainWindow.AllowTabbedDocks)

        self.setWindowTitle('疟疾检测')  # 设置标题
        self.setWindowIcon(QIcon('./UI/foreland.ico'))  # 设置标题图标
        self.bigBlackFont = QFont('', 14, QFont.Black)
        self.menu = self.menuBar()
        self.styleBTN = self.menu.addMenu("界面风格")
        self.genStyle()

        self.thinWeightslBTN = self.menu.addMenu("薄片模型")
        self.thickWeightslBTN = self.menu.addMenu("厚片模型")
        self.countryMenu = self.menu.addMenu("国家选择")  # 国家选择菜单
        self.gencontry()

        self.genWeights()

        self.openPicDirBTN = QPushButton('打开文件夹', self)
        self.takePhotoBTN = QPushButton('拍摄图片', self)
        self.thickBTN = QPushButton('厚度识别', self)
        self.modifyBTN = QPushButton("厚度修正", self)
        self.foregroundBTN = QPushButton('细胞检测', self)
        self.lastBTN = QPushButton('上一张', self)
        self.nextBTN = QPushButton('下一张', self)
        self.autoBTN = QPushButton('自动检测所有', self)
        self.stopButton = QPushButton("停止", self)
        self.exportBTN = QPushButton("导出", self)
        self.allInfoTE = QTextEdit()
        self.allInfoTE.setFocusPolicy(Qt.NoFocus)
        self.allInfoVL = QVBoxLayout(self.allInfoTE)
        self.infoTE = QTextEdit()  # 图片信息
        self.infoTE.setFocusPolicy(Qt.NoFocus)
        self.infoVL = QVBoxLayout(self.infoTE)
        self.outTE = QTextEdit()  # 细胞信息
        self.outTE.setFocusPolicy(Qt.NoFocus)
        self.outVL = QVBoxLayout(self.outTE)
        self.BTNWidget = QWidget()
        self.BTNVL = QVBoxLayout(self.BTNWidget)

        self.BTNVL.addWidget(self.openPicDirBTN)
        self.BTNVL.addWidget(self.takePhotoBTN)
        self.BTNVL.addWidget(self.thickBTN)
        self.BTNVL.addWidget(self.modifyBTN)
        self.BTNVL.addWidget(self.foregroundBTN)
        self.BTNVL.addWidget(self.lastBTN)
        self.BTNVL.addWidget(self.nextBTN)
        self.BTNVL.addWidget(self.autoBTN)
        self.BTNVL.addWidget(self.stopButton)
        self.BTNVL.addWidget(self.exportBTN)

        self.bugList = QListView()
        self.bugSim = QStandardItemModel()
        self.bugList.setModel(self.bugSim)
        self.bugList.setMinimumWidth(100)
        self.bugList.setSelectionMode(QAbstractItemView.MultiSelection)
        self.bugList.setIconSize(QSize(300, 300))
        self.bugList.clicked.connect(self.clickedBugList)

        self.bugList.setLayoutDirection(Qt.RightToLeft)
        self.bugList.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.fileTable = QTableView()
        self.fileSim = QStandardItemModel()
        self.fileTable.setModel(self.fileSim)
        self.fileTable.clicked.connect(self.clickedFileTable)
        self.fileTable.setMinimumWidth(100)

        self.fileTable.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.medicalRecordTable = QTableView()
        self.medicalRecordSim = QStandardItemModel()
        self.medicalRecordTable.setModel(self.medicalRecordSim)
        # self.medicalRecordTable.clicked.connect(self.clickedFileTable)
        self.medicalRecordTable.setMinimumWidth(100)
        self.medicalRecordTable.setEditTriggers(
            QAbstractItemView.NoEditTriggers)

        self.medicalRecordSim.setHorizontalHeaderLabels(['项目', '信息'])
        self.medicalRecordTable.verticalHeader().setVisible(False)

        items = [QStandardItem('姓名'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('国家'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('样本图片数量'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('样本厚片数量'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('样本薄片数量'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('样本细胞数量'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('样本虫体数量'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('样本白细胞数量'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('厚片平均感染度'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('薄片平均感染度'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('阴阳性'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('PF概率'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('PM概率'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('PO概率'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('PV概率'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)
        items = [QStandardItem('综合判断虫型为'), QStandardItem('')]
        items[0].setFont(self.bigBlackFont)
        items[1].setFont(self.bigBlackFont)
        self.medicalRecordSim.appendRow(items)

        self.medicalRecordTable.resizeColumnsToContents()
        # self.fileTable.setColumnWidth(2, 40)
        self.tableHeader = self.medicalRecordTable.horizontalHeader()
        self.tableHeader.setSectionResizeMode(QHeaderView.Stretch)

        self.viewers = {
            'raw': PhotoViewer(self),
            'cell': PhotoViewer(self),
        }
        self.viewers['cell'].itemClicked.connect(self.showCellInfo)

        self.cellDockWidget = QWidget()
        self.cellDockVL = QVBoxLayout(self.cellDockWidget)
        self.cellPB = QProgressBar(self.cellDockWidget)
        self.cellPB.setAlignment(Qt.AlignRight | Qt.AlignCenter)
        self.cellDockVL.addWidget(self.cellPB)
        self.cellDockVL.addWidget(self.viewers['cell'])

        self.bugDockWidget = QWidget()
        self.bugDockVL = QVBoxLayout(self.bugDockWidget)
        self.bugDockHLWidget = QWidget()
        self.bugDockHL = QHBoxLayout(self.bugDockHLWidget)
        self.allBugsBTN = QPushButton("全选", self)
        self.allBugsBTN.clicked.connect(self.selectAllBugs)
        self.exportBugsBTN = QPushButton("导出", self)
        self.exportBugsBTN.clicked.connect(self.exportPics)
        self.bugLabel = QLabel("")
        self.bugDockHL.addWidget(self.allBugsBTN)
        self.bugDockHL.addWidget(self.exportBugsBTN)
        self.bugDockHL.addWidget(self.bugLabel)
        self.bugDockVL.addWidget(self.bugList)
        self.bugDockVL.addWidget(self.bugDockHLWidget)

        self.docks = {
            'btn': self.getDock('操作', self.BTNWidget),
            'raw': self.getDock('原图', self.viewers['raw']),
            'cell': self.getDock('细胞', self.cellDockWidget),
            'files': self.getDock('图片地址', self.fileTable),
            'bug': self.getDock('疟原虫', self.bugDockWidget),
            'record': self.getDock('病历', self.medicalRecordTable),
            'info': self.getDock('图片信息', self.infoTE),
            'out': self.getDock('细胞信息', self.outTE),
            'allInfo': self.getDock('文件夹信息', self.allInfoTE),
        }
        self.addDockWidget(Qt.LeftDockWidgetArea, self.docks['btn'])
        self.splitDockWidget(self.docks['btn'], self.docks['record'],
                             Qt.Horizontal)
        self.splitDockWidget(self.docks['record'], self.docks['raw'],
                             Qt.Horizontal)
        self.splitDockWidget(self.docks['raw'], self.docks['bug'],
                             Qt.Horizontal)
        self.splitDockWidget(self.docks['bug'], self.docks['files'],
                             Qt.Horizontal)

        self.splitDockWidget(self.docks['btn'], self.docks['allInfo'],
                             Qt.Vertical)
        self.splitDockWidget(self.docks['record'], self.docks['info'],
                             Qt.Vertical)
        self.splitDockWidget(self.docks['info'], self.docks['out'],
                             Qt.Vertical)
        self.splitDockWidget(self.docks['raw'], self.docks['cell'],
                             Qt.Vertical)

        self.openPicDirBTN.clicked.connect(self.openPicDir)
        self.openPicDirBTN.setShortcut('ctrl+o')
        self.takePhotoBTN.clicked.connect(self.takePhoto)
        # self.takePhotoBTN.setShortcut('ctrl+p')
        self.foregroundBTN.clicked.connect(self.detectCells)
        self.foregroundBTN.setShortcut('f')
        self.exportBTN.clicked.connect(self.exportResult)
        self.exportBTN.setShortcut('s')
        self.lastBTN.clicked.connect(self.setLastPic)
        self.lastBTN.setShortcut('a')
        self.nextBTN.clicked.connect(self.setNextPic)
        self.nextBTN.setShortcut('d')
        self.thickBTN.clicked.connect(self.detectThick)
        self.thickBTN.setShortcut('l')
        self.autoBTN.clicked.connect(self.autoDetect)

        self.stopButton.setShortcut('q')
        self.stopButton.clicked.connect(self.stop_malariaThread)  # 连接信号和槽

        self.modifyBTN.setShortcut('m')
        self.modifyBTN.clicked.connect(self.modifyThickness)

        self.fileTable.resizeColumnsToContents()
        # self.fileTable.setColumnWidth(2, 40)
        self.tableHeader = self.fileTable.horizontalHeader()
        self.tableHeader.sectionClicked.connect(self.clickedFileTableHeader)
        # self.tableHeader.setSectionResizeMode(QHeaderView.Stretch)
        self.openPicDir(True)

        self.confidenceDialog.confidenceChanged.connect(self.handleConfidenceChange)
        self.confidenceDialog.valueCommitted.connect(self.handleValueCommitted)
        self.toggleSliderShortcut = QShortcut(Qt.Key_F2, self)
        self.toggleSliderShortcut.activated.connect(self.toggleConfidenceDialog)

    def gencontry(self):
        with open("./countries_list.txt", 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip().replace('"', '')
            qa = QAction(line, self)
            qa.setCheckable(True)
            qa.setChecked(False)
            qa.triggered.connect(lambda _, action=qa: self.onActionTriggered(action))
            self.countryMenu.addAction(qa)

    def onActionTriggered(self, action):
        # 遍历actions列表，取消选中所有其他QAction
        for a in self.countryMenu.actions():
            if a != action:
                a.setChecked(False)
        # 更新当前选中的国家
        self.current_country = action.text()
        # 更新菜单项的文本

        self.updateCountryMenuText()

    def updateCountryMenuText(self):
        # 更新“国家选择”菜单项的文本为当前选中的国家名称
        self.countryMenu.setTitle(self.current_country)
        # self.malariaThread = malariaThread(self.current_country)
        # self.malariaThread.load_weight("weights/thin-detection-8.16.pt", "weights/thick-00.pt")
        # self.malariaThread.trigger.connect(self.showOneCell)
        # self.malariaThread.auto.connect(self.detectCells)
        # self.malariaThread.results.connect(self.showResults)
        self.medicalRecordSim.item(1, 1).setText(str(self.current_country))
        # print('重新初始化线程')
        print(self.current_country)

    # def getDock(self, name, viewer):
    #     """生成dock窗口."""
    #     dock = QDockWidget(name, self)
    #     dock.setWidget(viewer)
    #     return dock

    def getDock(self, name, viewer):
        """生成dock窗口，隐藏标题栏以禁用复制功能。"""
        dock = QDockWidget(name, self)
        dock.setWidget(viewer)

        # 首先获取当前的特性
        current_features = dock.features()
        # 然后清除DockWidget所有的特性，复制和删除
        current_features &= ~QDockWidget.DockWidgetClosable
        current_features &= ~QDockWidget.DockWidgetFloatable
        # 然后更新dock的新的特性
        dock.setFeatures(current_features)
        return dock

    def showMat(self, which, bgr):
        """显示opencv图片."""
        if which not in self.viewers:
            print(which, 'not in viewer')
            return 0
        shrink = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # cv 图片转换成 qt图片
        qimg = QImage(shrink.data,  # 数据源
                      shrink.shape[1],  # 宽度
                      shrink.shape[0],  # 高度
                      shrink.shape[1] * 3,  # 行字节数
                      QImage.Format_RGB888)
        pimg = QPixmap.fromImage(qimg)
        self.viewers[which].setPhoto(pimg)

    def getOneBugItem(self, data):
        """展示单个细胞."""
        key, class_, conf, x, y, w, h = data

        shrink = cv2.cvtColor(self.mat['bgr'][y:y + h, x:x + w],
                              cv2.COLOR_BGR2RGB)
        # cv 图片转换成 qt图片
        qimg = QImage(shrink.data,  # 数据源
                      shrink.shape[1],  # 宽度
                      shrink.shape[0],  # 高度
                      shrink.shape[1] * 3,  # 行字节数
                      QImage.Format_RGB888)
        pimg = QPixmap.fromImage(qimg)
        bugItem = QStandardItem(str(key))
        bugItem.setCheckable(True)
        bugItem.setIcon(QIcon(pimg.scaled(QSize(200, 200))))
        return bugItem

    def selectAllBugs(self):
        """全选所有疟原虫图片."""
        total = self.bugSim.rowCount()
        if self.allBugsBTN.text() == "全选":
            self.bugLabel.setText(f"已选中 {total}/{total}")
            self.bugList.selectAll()
            self.allBugsBTN.setText("全不选")
            for row in range(total):
                self.bugSim.item(row, 0).setCheckState(True)
        elif self.bugList.selectedIndexes():
            self.bugLabel.setText(f"已选中 0/{total}")
            self.bugList.clearSelection()
            self.allBugsBTN.setText("全选")
            for row in range(total):
                self.bugSim.item(row, 0).setCheckState(False)

    def setPic(self):
        """读取一张图."""
        if self.picSeed < 0 or self.pics == []:
            return
        # print(self.pics[self.picSeed])
        self.curimg.load(self.pics[self.picSeed])
        self.mat['bgr'] = cvimread(self.pics[self.picSeed])
        self.mat['yolo'] = prefunc(self.mat['bgr'])
        self.showMat("raw", self.mat['bgr'])
        # self.viewers['raw'].setPhoto(self.curimg)
        self.infos[self.picSeed]['imageSize'] = str(self.mat['bgr'].shape)
        self.showPicInfo()
        self.docks['raw'].show()
        self.setWindowTitle(
            f'疟疾检测 {self.picSeed + 1}/{self.picNum} {self.pics[self.picSeed]}')
        print("图片seed={}\t 图片路径:{}".format(self.picSeed, self.pics[self.picSeed]))

    def detectThick(self, modify=False):
        """判断厚薄."""
        if self.picSeed < 0 or self.pics == []:
            return
        if self.infos[self.picSeed]["薄厚"]:
            res = [self.infos[self.picSeed]["薄厚"],
                   self.infos[self.picSeed]["薄厚置信度"]]
        else:
            res = self.thicknessClassifier.infer(self.mat['bgr'])
        if modify:
            if res[0] == 'thin':
                res[0] = 'thick'
            elif res[0] == 'thick':
                res[0] = 'thin'
        self.infos[self.picSeed]["薄厚"] = res[0]
        self.infos[self.picSeed]["薄厚置信度"] = res[1]
        self.showPicInfo()
        return res[0] == 'thin'

    def modifyThickness(self):
        """修正厚度识别结果."""
        self.detectThick(True)

    def detectCells(self):

        patient_status = '阴性'
        # self.autoBTN.setEnabled(False)
        if self.picSeed < 0 or self.pics == []:
            return
        """检测细胞."""
        self.bugSim.clear()
        self.viewers['cell'].clearBoxes()
        self.cellInfos[self.picSeed] = {}
        self.current_key = self.picSeed
        if self.current_key in self.detected_seed:
            print('detectCells检测细胞 finished, now try again!')
            self.cellPB.setValue(0)
            self.infos[self.picSeed]["细胞"] = {}
            # if self.cellThread.loop:
            #     self.setNextPic()
        self.detectThick(False)
        # self.cellThread.thickness = self.infos[self.picSeed]["薄厚"]

        # if thin 检测细胞、检测疟原虫
        # if thick 检测疟原虫
        key = 0
        cell_info_dict = {}
        px_info_dict = {}

        if self.infos[self.picSeed]["薄厚"] == 'thin':
            # 检测细胞
            self.pred_cell = self.cellDetecter.infer(self.mat['yolo'], device=DEVICE)
            for xywh, class_, conf in zip(self.pred_cell.boxes.xywh,
                                          self.pred_cell.boxes.cls,
                                          self.pred_cell.boxes.conf):
                # self.cellInfos[self.picSeed][key] = {}
                cell_info_dict[key] = {}
                cx, cy, w, h = map(int, xywh)
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                # if "cell" not in self.infos[self.picSeed]["细胞"]:
                #     self.infos[self.picSeed]["细胞"]["cell"] = 0
                # else:
                #     self.infos[self.picSeed]["细胞"]["cell"] += 1
                # self.cellInfos[self.picSeed][key]['类型'] = "cell"
                # self.cellInfos[self.picSeed][key]['置信度'] = conf.item()
                # self.cellInfos[self.picSeed][key]['位置矩形'] = (x, y, w, h)
                cell_info_dict[key]['类型'] = "cell"
                cell_info_dict[key]['置信度'] = conf.item()
                cell_info_dict[key]['位置矩形'] = (x, y, w, h)
                key += 1
        # 检测疟原虫
        # result_malaria = self.malariaThread.run_once(self.mat['bgr'], self.infos[self.picSeed]["薄厚"])

        result_malaria, conf_numbers, file_numbers = self.malariaThread.run_once(self.pics[self.picSeed],
                                                                                 self.infos[self.picSeed]["薄厚"])
        # print(result_malaria)
        '''
        result_malaria
        
        '''

        for xywh, class_, conf in zip(result_malaria.boxes.xywh,
                                      result_malaria.boxes.cls,
                                      result_malaria.boxes.conf):
            # self.cellInfos[self.picSeed][key] = {}
            # print(class_)
            px_info_dict[key] = {}

            class_ = result_malaria.names[int(class_)].lower()
            cx, cy, w, h = map(int, xywh)
            x = int(cx - w / 2)
            y = int(cy - h / 2)
            # self.cellInfos[self.picSeed][key]['类型'] = "cell"
            # self.cellInfos[self.picSeed][key]['置信度'] = conf.item()
            # self.cellInfos[self.picSeed][key]['位置矩形'] = (x, y, w, h)

            # px_info_dict[key]['类型'] = class_

            if class_ in ['pv', 'pm', 'po', 'pf']:
                class_ = 'malaria'

            # 此处强行将细分  改为  疟原虫
            px_info_dict[key]['类型'] = class_
            px_info_dict[key]['置信度'] = conf.item()
            px_info_dict[key]['位置矩形'] = (x, y, w, h)

            key += 1

        # 合并细胞和疟原虫信息,计算交并比

        # if self.picSeed not in self.detected_seed:
        #     self.detected_seed.append(self.picSeed)
        #
        # # print(cell_info_dict)
        # # print(px_info_dict)
        # # 对cell和Px分别处理
        # max_index = 0
        # for cckey, ccvalue in cell_info_dict.items():
        #     self.cellInfos[self.picSeed][cckey] = ccvalue
        #     max_index = cckey + 1
        #     for ppkey, ppvalue in px_info_dict.items():
        #         if compute_IOU(ccvalue["位置矩形"], ppvalue["位置矩形"]) >= 0.5:
        #             self.cellInfos[self.picSeed][cckey] = ppvalue
        #             del px_info_dict[ppkey]
        #             break
        # for pppkey, pppvalue in px_info_dict.items():
        #     self.cellInfos[self.picSeed][max_index] = pppvalue
        #
        # """逐个检测展示细胞."""
        # # self.cellPB.setRange(0, len(self.cellInfos[self.picSeed]))
        # for infokey, infovalue in self.cellInfos[self.picSeed].items():
        #     key = infokey
        #     class_ = infovalue["类型"]
        #     conf = infovalue["置信度"]
        #     x, y, w, h = infovalue["位置矩形"]
        #     if class_ not in self.infos[self.picSeed]["细胞"]:
        #         self.infos[self.picSeed]["细胞"][class_] = 0
        #     self.infos[self.picSeed]["细胞"][class_] += 1
        #     # self.cellPB.setValue(key + 1)
        #     # self.cellPB.setFormat(str(key))
        #     qcolor = self.qcolors[class_.lower()]
        #     self.viewers['cell'].addBox(class_, x, y, w, h, qcolor)
        #     if class_ in ["wbc", "po", "pf", "pv", "pv-s", "pm", 'malaria']:
        #         data = [key, class_, conf, x, y, w, h]
        #         self.bugSim.appendRow(self.getOneBugItem(data))

        # 合并细胞和疟原虫信息
        if self.picSeed not in self.detected_seed:
            self.detected_seed.append(self.picSeed)

        # 合并细胞和寄生虫信息
        combined_info = {**cell_info_dict, **px_info_dict}
        self.cellInfos[self.picSeed] = combined_info

        # 更新统计信息
        for infokey, infovalue in self.cellInfos[self.picSeed].items():
            class_ = infovalue["类型"]
            conf = infovalue["置信度"]
            x, y, w, h = infovalue["位置矩形"]
            if class_ not in self.infos[self.picSeed]["细胞"]:
                self.infos[self.picSeed]["细胞"][class_] = 0
            self.infos[self.picSeed]["细胞"][class_] += 1

            # 可视化细胞和寄生虫的位置
            qcolor = self.qcolors[class_.lower()]
            self.viewers['cell'].addBox(class_, x, y, w, h, qcolor)
            if class_ in ["wbc", "po", "pf", "pv", "pv-s", "pm", 'malaria']:
                data = [infokey, class_, conf, x, y, w, h]
                self.bugSim.appendRow(self.getOneBugItem(data))

        if file_numbers >= 1 or conf_numbers >= 1:
            patient_status = '阳性'
        self.infos[self.picSeed]['检测结果'] = patient_status

        self.medicalRecordSim.item(10, 1).setText(str(patient_status))

        self.showPicInfo(False)

    def showResults(self, data):

        self.data = data
        # patient_status = data[0]
        # 图片信息展示阴阳和虫型
        # self.infos[self.picSeed]['检测结果'] = data[0]
        # self.infos[self.picSeed]['检测虫型'] = data[1]

        max_key = self.data[1]
        country_list_PV = ['埃塞俄比亚', '阿斯马拉', '厄立特里亚', '吉布提', '索马里半岛', '东南亚', '南亚', '拉丁美洲',
                           '北苏丹']
        country_list_PO = [
            '南苏丹', '乍得', '中非', '喀麦隆', '赤道几内亚', '加蓬', '刚果共和国（即刚果(布)）',
            '刚果民主共和国（即刚果(金)）',
            '圣多美及普林西比',
            '毛里塔尼亚', '西撒哈拉', '塞内加尔', '冈比亚', '马里', '布基纳法索', '几内亚', '几内亚比绍', '佛得角',
            '塞拉利昂', '利比里亚',
            '科特迪瓦', '加纳', '多哥', '贝宁', '尼日尔', '赞比亚', '安哥拉', '津巴布韦', '马拉维', '莫桑比克',
            '博茨瓦纳', '纳米比亚',
            '南非', '斯威士兰', '莱索托', '马达加斯加', '科摩罗', '毛里求斯', '留尼旺（法）', '圣赫勒拿（英）',
            '肯尼亚', '坦桑尼亚', '乌干达', '卢旺达', '布隆迪', '塞舌尔'
        ]
        country_list_NONE = ['未知国家', '选择国家']
        print(self.current_country)
        print(max_key)
        if self.current_country in country_list_PV and (max_key == 'PO' or max_key == 'PV'):
            max_key = 'PV'
            self.data[1] = max_key

        elif self.current_country in country_list_PO and (max_key == 'PO' or max_key == 'PV'):
            max_key = 'PO'
            self.data[1] = max_key

        elif self.current_country in country_list_NONE:
            self.data = data
        else:
            QMessageBox.warning(self, '警告', '当前国家未在划分目录名单，请仔细核对或联系工作人员')

        # self.medicalRecordSim.item(10, 1).setText(str(self.data[0]))
        # self.medicalRecordSim.item(11, 1).setText(str(self.data[2]))
        # self.medicalRecordSim.item(12, 1).setText(str(self.data[3]))
        # self.medicalRecordSim.item(13, 1).setText(str(self.data[4]))
        # self.medicalRecordSim.item(14, 1).setText(str(self.data[5]))
        # self.medicalRecordSim.item(15, 1).setText(str(self.data[1]))

        self.showPicInfo(False)

    def showOneCell(self, data):
        """批处理后展示细胞."""
        self.viewers['cell'].clearBoxes()

        self.picSeed = data["imgseed"]
        self.infos[self.picSeed]["细胞"] = {}
        self.cellInfos[self.picSeed] = {}
        self.setPic()
        # self.detectThick(False)
        self.infos[self.picSeed]["薄厚"] = data["thinkness"]
        self.infos[self.picSeed]["薄厚置信度"] = data["thinkness_conf"]

        if self.picSeed not in self.detected_seed:
            self.detected_seed.append(self.picSeed)
        for infokey, infovalue in data["info"].items():
            key = infokey
            class_ = infovalue["类型"]
            conf = infovalue["置信度"]
            x, y, w, h = infovalue["位置矩形"]
            self.cellInfos[self.picSeed][key] = infovalue
            if class_ not in self.infos[self.picSeed]["细胞"]:
                self.infos[self.picSeed]["细胞"][class_] = 0
            self.infos[self.picSeed]["细胞"][class_] += 1
            qcolor = self.qcolors[class_.lower()]
            self.viewers['cell'].addBox(class_, x, y, w, h, qcolor)
            if class_ in ["wbc", "po", "pf", "pv", "pv-s", "pm", 'malaria']:
                data = [key, class_, conf, x, y, w, h]
                self.bugSim.appendRow(self.getOneBugItem(data))
        self.showPicInfo(False)

    def showPicInfo(self, showCellsFlag=True):
        """展示图片信息."""
        self.viewers['raw'].setInfo(self.infos[self.picSeed]["姓名"])
        if self.infos[self.picSeed]["薄厚"] == 'thin':
            self.infos[self.picSeed]["模型"] = os.path.basename(
                self.malariaThread.thin_weight_path)

            self.infos[self.picSeed]['感染度'] = 0

            if 'wbc' in self.infos[self.picSeed]["细胞"] or 'po' in self.infos[self.picSeed]["细胞"] or 'pf' in \
                    self.infos[self.picSeed]["细胞"] or 'pv' in self.infos[self.picSeed]["细胞"] or 'pv-s' in \
                    self.infos[self.picSeed]["细胞"] or 'pm' in self.infos[self.picSeed]["细胞"]:
                parasites = 0
                for key, value in self.infos[self.picSeed]["细胞"].items():
                    if key != "cell":
                        parasites += value

                if 'cell' in self.infos[self.picSeed]["细胞"]:
                    total = self.infos[self.picSeed]["细胞"]["cell"]
                else:
                    total = sum(self.infos[self.picSeed]["细胞"].values())
                if total <= 0:
                    self.infos[self.picSeed]['感染度'] = 0
                else:
                    self.infos[self.picSeed]['感染度'] = round(
                        parasites / total * 100, 2)
        elif self.infos[self.picSeed]["薄厚"] == 'thick':
            self.infos[self.picSeed]["模型"] = os.path.basename(
                self.malariaThread.thick_weight_path)
            if 'wbc' in self.infos[self.picSeed]["细胞"] or 'po' in self.infos[self.picSeed]["细胞"] or 'pf' in \
                    self.infos[self.picSeed]["细胞"] or 'pv' in self.infos[self.picSeed]["细胞"] or 'pv-s' in \
                    self.infos[self.picSeed]["细胞"] or 'pm' in self.infos[self.picSeed]["细胞"]:
                malaria = 0
                for key, value in self.infos[self.picSeed]["细胞"].items():
                    if key != "cell" and key != "wbc":
                        malaria += value
                if "wbc" in self.infos[self.picSeed]["细胞"]:
                    WBC = self.infos[self.picSeed]["细胞"]['wbc']
                    if WBC == 0:
                        self.infos[self.picSeed]['感染度'] = 0
                    else:
                        self.infos[self.picSeed]['感染度'] = round(
                            malaria / WBC * 8000)
                else:
                    self.infos[self.picSeed]['感染度'] = 0

        self.infoTE.setText(json.dumps(self.infos[self.picSeed], indent=4,
                                       ensure_ascii=False))

        allInfo = {}
        for key in self.infos:
            for class_ in self.infos[key]["细胞"]:
                if class_ not in allInfo:
                    allInfo[class_] = 0
                allInfo[class_] += int(self.infos[key]["细胞"][class_] > 0)

        # show = {}
        # for key in allInfo:
        #     show["pic with " + key] = allInfo[key]
        show = {}
        for key in allInfo:
            show["文件夹名称"] = self.picDir  # 使用新的键名称添加到 show 字典中
            if key == "cell":  # 检查键的名称是否为 "pic with cell"
                new_key = "当前已检测图片数"  # 设置新的键名称
            else:
                new_key = key + "的图片数量 "  # 对于其他键，保持原有的格式
            show[new_key] = allInfo[key]  # 使用新的键名称添加到 show 字典中

        self.allInfoTE.setText(json.dumps(show, indent=4,
                                          ensure_ascii=False))

        self.clickedBugList()
        if showCellsFlag:
            self.showCells()
        else:
            for i, key in enumerate(self.cellClasses):
                c1 = str(self.infos[self.picSeed]
                         ["细胞"].get(key.lower(), 0))
                self.fileSim.item(self.picSeed, i + 1).setText(c1)

        people = self.infos[self.picSeed]["姓名"]
        self.medicalRecordSim.item(0, 1).setText(people)
        self.medicalRecordSim.item(1, 1).setText('地区未知')
        self.medicalRecordSim.item(2, 1).setText(
            str(len(self.peopleInfo[people]["pics"])))
        self.peopleInfo[people]["score"][self.picSeed] = \
            self.infos[self.picSeed]['感染度']

        thinScore, thickScore = [], []
        thinNum, thickNum = 0, 0
        bugNum, cellNum, wbcNum = 0, 0, 0
        for seed in self.peopleInfo[people]["score"]:
            if self.infos[seed]["薄厚"] == 'thin':
                thinScore.append(self.peopleInfo[people]["score"][seed])
                thinNum += 1

                for key in self.cellInfos[seed]:
                    if "类型" not in self.cellInfos[seed][key]:
                        continue
                    class_ = self.cellInfos[seed][key]['类型']
                    # if class_ in ["wbc", "po", "pf", "pv", "pv-s", "pm", 'malaria']:

                    if class_ in ["po", "pf", "pv", "pv-s", "pm", 'malaria']:
                        bugNum += 1
                    elif class_ == "wbc":
                        wbcNum += 1
                    if "cell" not in self.infos[self.picSeed]["细胞"]:
                        cellNum += 1
                    elif class_ == "cell":
                        cellNum += 1
            elif self.infos[seed]["薄厚"] == 'thick':
                thickScore.append(self.peopleInfo[people]["score"][seed])
                thickNum += 1
                for key in self.cellInfos[seed]:
                    if "类型" not in self.cellInfos[seed][key]:
                        continue
                    class_ = self.cellInfos[seed][key]['类型']
                    if class_ in ["po", "pf", "pv", "pv-s", "pm", 'malaria']:
                        bugNum += 1
                    elif class_ == "wbc":
                        wbcNum += 1




        if thinScore:
            thinScore = sum(thinScore) / len(thinScore)
        else:
            thinScore = 0
        if thickScore:
            thickScore = sum(thickScore) / len(thickScore)
        else:
            thickScore = 0

        self.medicalRecordSim.item(1, 1).setText(str(self.current_country))

        self.medicalRecordSim.item(3, 1).setText(str(thickNum))
        self.medicalRecordSim.item(4, 1).setText(str(thinNum))
        self.medicalRecordSim.item(5, 1).setText(str(cellNum))
        self.medicalRecordSim.item(6, 1).setText(str(bugNum))
        self.medicalRecordSim.item(7, 1).setText(str(wbcNum))
        self.medicalRecordSim.item(8, 1).setText(str(round(thickScore, 2)))
        self.medicalRecordSim.item(9, 1).setText(str(round(thinScore, 2)))






        #   更换图片时，更新数据
        if self.data:
            self.medicalRecordSim.item(10, 1).setText(str(self.data[0]))
            self.medicalRecordSim.item(11, 1).setText(str(self.data[2]))
            self.medicalRecordSim.item(12, 1).setText(str(self.data[3]))
            self.medicalRecordSim.item(13, 1).setText(str(self.data[4]))
            self.medicalRecordSim.item(14, 1).setText(str(self.data[5]))
            self.medicalRecordSim.item(15, 1).setText(str(self.data[1]))
        else:
            self.medicalRecordSim.item(10, 1).setText(str(0))
            self.medicalRecordSim.item(11, 1).setText(str(0))
            self.medicalRecordSim.item(12, 1).setText(str(0))
            self.medicalRecordSim.item(13, 1).setText(str(0))
            self.medicalRecordSim.item(14, 1).setText(str(0))
            self.medicalRecordSim.item(15, 1).setText(str(0))

        if self.infos[seed]["薄厚"] == 'thick':
            if bugNum > 5:
                self.medicalRecordSim.item(10, 1).setText(str("阳性"))

        # 更新进度条
        if self.malariaThread.isRunning():
            self.cellPB.setRange(0, self.picNum)
            print(self.picNum)
            print(self.picSeed)

            self.cellPB.setValue(self.picSeed + 1)
            self.cellPB.setFormat(str(self.picSeed))

    def setLastPic(self):
        """上一张图."""
        self.picSeed -= 1
        if self.picSeed < 0:
            self.picSeed = self.picNum - 1
        self.outTE.clear()
        self.setPic()

    def setNextPic(self):
        """下一张图."""
        self.picSeed += 1
        if self.picSeed >= self.picNum:
            self.picSeed = 0
        self.outTE.clear()
        self.setPic()

    def autoDetect(self):
        if self.auto:  # 默认路径则为False
            QMessageBox.warning(self, '警告', '当前为默认路径，请选择有效路径')
            return
        self.foregroundBTN.setEnabled(False)
        self.openPicDirBTN.setEnabled(False)
        if self.picSeed < 0 or self.pics == []:
            return

        """自动检测所有."""
        # self.openPicDirBTN.setEnabled(self.cellThread.loop)
        # self.takePhotoBTN.setEnabled(self.cellThread.loop)
        # self.foregroundBTN.setEnabled(self.cellThread.loop)
        # self.exportBTN.setEnabled(self.cellThread.loop)
        # self.thickBTN.setEnabled(self.cellThread.loop)
        # self.lastBTN.setEnabled(self.cellThread.loop)
        # self.nextBTN.setEnabled(self.cellThread.loop)
        # self.thinWeightslBTN.setEnabled(self.cellThread.loop)
        # self.thickWeightslBTN.setEnabled(self.cellThread.loop)
        # self.modifyBTN.setEnabled(self.cellThread.loop)
        # self.cellThread.loop = not self.cellThread.loop
        # self.detectCells()
        # 先清除缓存

        self.bugSim.clear()
        # self.fileSim.clear()
        self.outTE.clear()
        self.viewers['cell'].clearBoxes()
        # 设置相关参数
        self.detectThick(False)
        self.malariaThread.conf = self.confidenceDialog.currentValue / 100
        self.malariaThread.thinkness = self.infos[self.picSeed]["薄厚"]
        self.malariaThread.batch_path = self.picDir

        # self.malariaThread.run_once(self.picDir,self.infos[self.picSeed]["薄厚"])

        self.malariaThread.finished.connect(self.on_thread_finished)
        self.malariaThread.start()
        # self.malariaThread.run()

    def stop_malariaThread(self):
        if self.malariaThread.isRunning():
            self.malariaThread.terminate()
            self.malariaThread.wait()  # 等待线程安全退出
            self.foregroundBTN.setEnabled(True)  # 恢复按钮可用性
            self.openPicDirBTN.setEnabled(True)  # 恢复按钮可用性

    def on_thread_finished(self):
        # 任务完成后，重新启用锁定按钮
        self.foregroundBTN.setEnabled(True)
        self.openPicDirBTN.setEnabled(True)

    def showForeground(self):
        """展示前景."""
        fore = findCell(self.mat['bgr']).getForeground()
        self.showMat('cell', fore)
        self.docks['cell'].show()

    def exportResult(self):
        """导出数据."""
        path = QFileDialog.getExistingDirectory(self, "保存文件夹")
        with open(os.path.join(path, "picsInfo.json"), 'w') as f:
            f.write(json.dumps(self.infos, indent=4, ensure_ascii=False))

        with open(os.path.join(path, "cellsInfo.pkl"), 'wb') as f:
            pickle.dump(self.cellInfos, f)

    def exportPics(self):
        """保存被感染细胞."""
        root = QFileDialog.getExistingDirectory(self, "保存文件夹")
        ids = self.bugList.selectedIndexes()
        keys = []
        for qModelIndex in ids:
            row = qModelIndex.row()
            state = self.bugSim.item(row, 0).text()
            keys.append(int(state))
        for key in keys:
            if "类型" not in self.cellInfos[self.picSeed][key]:
                continue

            class_ = self.cellInfos[self.picSeed][key]["类型"]
            x, y, w, h = self.cellInfos[self.picSeed][key]['位置矩形']
            if class_ in ["wbc", "po", "pf", "pv", "pv-s", "pm", 'malaria']:
                img = self.mat['bgr'][y:y + h, x:x + w]
                name = os.path.basename(self.pics[self.picSeed])
                subName = name.replace('.', f"_{key}.")
                path = os.path.join(root, subName)
                cvimwrite(path, img)

    def showCells(self):
        """定位细胞."""
        self.showMat('cell', self.mat['bgr'])
        self.viewers['cell'].clearBoxes()
        self.bugSim.clear()
        if self.picSeed not in self.cellInfos:
            self.cellInfos[self.picSeed] = {}
        for key in self.cellInfos[self.picSeed]:

            if "类型" not in self.cellInfos[self.picSeed][key]:
                continue
            else:
                class_ = self.cellInfos[self.picSeed][key]["类型"]
            qcolor = self.qcolors[class_.lower()]
            x, y, w, h = self.cellInfos[self.picSeed][key]['位置矩形']
            self.viewers['cell'].addBox(class_, x, y, w, h, qcolor)
            if class_ in ["wbc", "po", "pf", "pv", "pv-s", "pm", 'malaria']:
                conf = self.cellInfos[self.picSeed][key]['置信度']
                data = [key, class_, conf, x, y, w, h]
                self.bugSim.appendRow(self.getOneBugItem(data))

    def showCellInfo(self, data):
        """展示细胞信息."""
        index, name = data
        if index < 0:
            return
        info = self.cellInfos[self.picSeed][index].copy()
        info['位置矩形'] = str(info['位置矩形'])
        info["id"] = index

        self.outTE.setText(json.dumps(info, indent=4,
                                      ensure_ascii=False))

    def clickedFileTable(self, qModelIndex):

        """切换图片."""
        row = qModelIndex.row()
        col = qModelIndex.column()
        folder_path = os.path.dirname(self.picDir)
        self.outTE.clear()
        print(folder_path)
        if self.auto:  # 默认路径则为False
            QMessageBox.warning(self, '警告', '当前为默认路径，请选择有效路径')
            return

        if col == 0:
            self.picSeed = row
            self.setPic()

    def clickedFileTableHeader(self, index):
        """自动调整列宽."""
        self.fileTable.resizeColumnToContents(index)

    def clickedBugList(self, qModelIndex=None):
        """选中一张疟原虫图片."""
        ids = self.bugList.selectedIndexes()

        total = self.bugSim.rowCount()
        self.bugLabel.setText(f"已选中{len(ids)}/{total}")
        if not (qModelIndex is None):
            row = qModelIndex.row()
            col = qModelIndex.column()
            state = self.bugSim.item(row, col).checkState()
            self.bugSim.item(row, col).setCheckState(not state)

    def openPicDir(self, auto=False):
        """打开图片文件."""
        self.initInfo()
        self.auto = auto
        if not auto:
            self.data = []
            # self.picDir = QFileDialog.getExistingDirectory(self, '打开图片文件夹')
            # 用户选择目录
            self.picDir = QFileDialog.getExistingDirectory(self, '打开图片文件夹')
            self.cellPB.setValue(0)
            # 检查用户是否取消了选择
            if not self.picDir:  # 如果路径为空，即用户没有选择目录
                QMessageBox.warning(self, '警告', '当前未选择有效路径，已重新加载默认路径')
                self.picDir = r'./dataset/PO_TEST'
                # return  # 退出函数，提示用户重新选择

            self.outTE.clear()
        else:
            self.picDir = r'./dataset/PO_TEST'
        self.picDir = os.path.normpath(self.picDir)
        if os.path.exists(self.picDir):
            self.pics.clear()
            for a, b, c in os.walk(self.picDir):
                for p in c:
                    path = os.path.normpath(os.path.join(a, p))
                    extension = filetype.guess_extension(path)
                    if extension == "jpg" or extension == "png":
                        self.pics.append(path)

        self.pics = sorted(set(self.pics))
        self.picNum = len(self.pics)
        for i in range(self.picNum):
            picName = os.path.split(self.pics[i])[1]
            if '-' in picName:
                people = picName.split('-')[0]
            else:
                people = ''
            self.infos[i] = infoDict(self.pics[i], people)
            if people not in self.peopleInfo:
                self.peopleInfo[people] = {'pics': [], "score": {}}
            self.peopleInfo[people]["pics"].append(i)

        if self.picNum:
            self.picSeed = 0
            self.setPic()
        else:
            self.picSeed = -1

        self.fileSim.clear()
        self.fileSim.setHorizontalHeaderLabels(
            ['地址'] + self.cellClasses)
        for i, p in enumerate(self.pics):
            items = [QStandardItem(p)] + [QStandardItem('0')
                                          for c in self.cellClasses]
            items[0].setCheckable(True)
            self.fileSim.appendRow(items)
        self.viewers['cell'].clearBoxes()
        self.bugSim.clear()

    def takePhoto(self):
        """拍摄图片"""
        # 拍照存放在self.picDir，然后做openPicDir后面的操作(遍历文件夹获得每张图片的信息，显示在listview中等等)

        pass

    def genStyle(self):
        """生成下拉切换风格的菜单."""
        self.qsses = {}

        def fun(qss):
            def gen():
                for k in self.qsses:
                    self.qsses[k].setChecked(False)
                self.qsses[qss].setChecked(True)

                self.setStyleSheet(open(f"./UI/QSS-master/{qss}").read())

            return gen

        for qss in os.listdir("./UI/QSS-master"):
            if qss.endswith(".qss"):
                qa = QAction(qss[:-4], self, checkable=True)
                qa.setChecked(False)
                self.qsses[qss] = qa
                qa.triggered.connect(fun(qss))
                self.styleBTN.addAction(qa)

        from qt_material import apply_stylesheet, list_themes

        def fun(qss):
            def gen():
                for k in self.qsses:
                    self.qsses[k].setChecked(False)
                self.qsses[qss].setChecked(True)
                apply_stylesheet(self, theme=qss)

            return gen

        for qss in list_themes():
            qa = QAction(qss, self, checkable=True)
            qa.setChecked(False)
            self.qsses[qss] = qa
            qa.triggered.connect(fun(qss))
            self.styleBTN.addAction(qa)
        apply_stylesheet(self, theme="light_teal_500.xml")

    def genWeights(self):
        """生成选择薄片模型的菜单."""
        self.thinWeights = {}
        self.thickWeights = {}
        self.currentThinModel = ""
        self.currentThickModel = ""

        # def thinFun(file):
        #     def gen():
        #         for k in self.thinWeights:
        #             self.thinWeights[k].setChecked(False)
        #         self.thinWeights[file].setChecked(True)
        #         if 'thin' in file:
        #             self.malariaThread.load_thin_weight(f"weights/{file}")
        #
        #     return gen
        #
        # def thickFun(file):
        #     def gen():
        #         for k in self.thickWeights:
        #             self.thickWeights[k].setChecked(False)
        #         self.thickWeights[file].setChecked(True)
        #         self.malariaThread.load_thick_weight(f"weights/{file}")
        #
        #     return gen
        #
        # for model in os.listdir("./weights"):
        #     if model.endswith(".pt"):
        #         qa = QAction(model, self, checkable=True)
        #         qa.setChecked(False)
        #         if model.startswith("thin"):
        #             self.thinWeights[model] = qa
        #             qa.triggered.connect(thinFun(model))
        #             self.thinWeightslBTN.addAction(qa)
        #         elif model.startswith("thick"):
        #             self.thickWeights[model] = qa
        #             qa.triggered.connect(thickFun(model))
        #             self.thickWeightslBTN.addAction(qa)

        def thinFun(file):
            def gen():
                self.currentThinModel = file  # 保存当前选择的薄片模型名称
                for k in self.thinWeights:
                    self.thinWeights[k].setChecked(False)
                self.thinWeights[file].setChecked(True)
                self.malariaThread.load_thin_weight(f"weights/{file}")

            return gen

        def thickFun(file):
            def gen():
                self.currentThickModel = file  # 保存当前选择的厚片模型名称
                for k in self.thickWeights:
                    self.thickWeights[k].setChecked(False)
                self.thickWeights[file].setChecked(True)
                self.malariaThread.load_thick_weight(f"weights/{file}")

            return gen

        for model in os.listdir("./weights"):
            if model.endswith(".pt"):
                qa = QAction(model, self, checkable=True)
                qa.setChecked(False)
                if model.startswith("thin"):
                    self.thinWeights[model] = qa
                    qa.triggered.connect(thinFun(model))
                    self.thinWeightslBTN.addAction(qa)
                elif model.startswith("thick"):
                    self.thickWeights[model] = qa
                    qa.triggered.connect(thickFun(model))
                    self.thickWeightslBTN.addAction(qa)

    def toggleConfidenceDialog(self):
        self.confidenceDialog.show()

    def handleConfidenceChange(self, value):
        # self.malariaThread.conf = value / 100
        print(f"Confidence changed to: {value}%")

    def handleValueCommitted(self, value):
        # 处理对话框关闭时的逻辑，例如保存值
        self.malariaThread.conf = value / 100
        print(f"Value committed: {value}%")
        print(f"Confidence changed to: {self.malariaThread.conf}%")

        # 重新加载当前选择的模型
        if self.currentThinModel:
            self.malariaThread.load_thin_weight(f"weights/{self.currentThinModel}")
        elif self.currentThickModel:
            self.malariaThread.load_thick_weight(f"weights/{self.currentThickModel}")

    def closeEvent(self, event):
        """关闭窗口."""
        # self.cellThread.terminate()
        self.malariaThread.terminate()


if __name__ == "__main__":
    # 创建一个app程序
    start = time.time()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.initUI()
    win.loadWeight()
    # win.showFullScreen()
    win.showMaximized()
    sys.exit(app.exec_())
