# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:38:33 2023.

@author: szh1213
"""


import cv2
import os
import numpy as np


def infoDict(path: str = '', people: str = '') -> dict:
    """信息结构."""
    info = {
        "薄厚": '',
        "薄厚置信度": -1,
        "图片路径": path,
        "姓名": people,
        "感染度": -1,
        "细胞": {}
    }

    return info


def cvimread(path):
    """读取中文路径图片."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


def cvimwrite(path, img):
    """保存中文路径图片."""
    path = os.path.abspath(path)
    cat = os.path.split(path)[0]
    if not os.path.exists(cat):
        os.makedirs(cat)
    c = '.'+path.split('.')[-1]
    cv2.imencode(c, img)[1].tofile(path)


class findCell:
    """
    查找细胞.

    查找染色
    """

    def __init__(self, img: np.array):
        if len(img.shape) == 2:
            self.bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            self.bgr = img

    def getForeground(self):
        """
        获取前景.

        Returns
        -------
        img : TYPE
            DESCRIPTION.

        """
        bgrLow = np.array([1, 1, 1], dtype=np.uint8)
        bgrHigh = np.array([255, 255, 220], dtype=np.uint8)
        mask = cv2.inRange(self.bgr, bgrLow, bgrHigh)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel)
        mask = cv2.erode(mask, kernel)

        paper = cv2.bitwise_and(self.bgr, self.bgr, mask=mask)
        self.paperMask = mask

        hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        hsvMask = h.copy()
        hsvMask[h < 90] = 0
        img = cv2.bitwise_and(self.bgr, self.bgr, mask=hsvMask)
        return img

    def getStained(self, img=None):
        if img is None:
            img = self.bgr

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        b, g, r = cv2.split(self.bgr)
        mask = s.copy()
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel)
        mask = cv2.erode(mask, kernel)
        mask[s < 70] = 0
        mask[g > 50] = 0
        cvimwrite('../output/h.jpg', h)
        cvimwrite('../output/s.jpg', s)
        cvimwrite('../output/v.jpg', v)
        cvimwrite('../output/b.jpg', b)
        cvimwrite('../output/g.jpg', g)
        cvimwrite('../output/r.jpg', r)
        # hsvLow = np.array([1, 0, 1],dtype=np.uint8)
        # hsvHigh = np.array([255, 255, 90],dtype=np.uint8)

        # hsvMask = cv2.inRange(hsv, hsvLow, hsvHigh)
        # if hsvMask.sum()/255 < 10000:
        #     hsvMask = np.zeros_like(hsvMask, dtype=np.uint8)
        # kernel = np.ones((5, 5), np.uint8)
        # hsvMask = cv2.erode(hsvMask, kernel)
        # kernel = np.ones((280, 280), np.uint8)
        # hsvMask = cv2.dilate(hsvMask, kernel)

        # retval, labels, stats, centroids = cv2.connectedComponentsWithStats(hsvMask, connectivity=8)
        # for i in range(retval):
        #     if stats[i,2]*stats[i,3]>self.bgr.shape[0]*self.bgr.shape[1]*0.95:
        #         stats[i][4]=0
        # cellIndex = stats[:,4].argmax()
        # hsvMask[labels!=cellIndex] = 0
        img = cv2.bitwise_and(img, img, mask=mask)
        # self.cellMask = hsvMask
        return img

    def haveBlack(self):
        h, w = self.bgr.shape[:2]
        a = self.bgr[[0, h-1], :, :]
        b = self.bgr[:, [0, w-1], :]
        a = cv2.transpose(a)
        c = cv2.vconcat((a, b))
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        bright = gray.mean()

        print('\n', bright)
        return bright < 30


if __name__ == '__main__':
    path = r'D:\code\yolov8\datasets\seg\images\付建国-0007.jpg'
    img = cvimread(path)
    model = findCell(img)
    st = model.getStained()
    cvimwrite('../output/st.jpg', st)
