# -*- coding: utf-8 -*-
"""
opencv相关算法.

模块负责人： .
"""

import cv2
import numpy as np

__version__ = '1.0.0'


def distinguish(img: np.ndarray, name: str) -> np.ndarray:
    """
    判断标本的薄厚程度.

    Parameters.

    ----------
    img : np.ndarray
        BGR格式图像.
    name : str
        图片文件名.

    Returns
    -------
    resultImg : np.ndarray
        处理的结果.
    thickness : float(0,1)
        厚为1薄为0，异常为负数.

    """
    resultImg = None
    thickness = -1
    return resultImg, thickness


def getForegroundMask(img: np.ndarray, name: str) -> np.ndarry:
    """
    获取图像前景mask,包括所有细胞、虫体、脏污.

    Parameters.

    ----------
    img : np.ndarray
        BGR格式图像.
    name : str
        图片文件名.

    Returns
    -------
    foregroundMask : np.ndarray
        opencv的单通道二值图.

    """
    foregroundMask = None
    return foregroundMask


def getStainedCellsMask(img: np.ndarray, name: str) -> np.ndarry:
    """
    获取图像中所有被染色的细胞mask.

    Parameters.

    ----------
    img : np.ndarray
        BGR格式图像.
    name : str
        图片文件名.

    Returns
    -------
    stainedCellsMask : np.ndarray
        opencv的单通道二值图.

    """
    stainedCellsMask = None
    return stainedCellsMask
