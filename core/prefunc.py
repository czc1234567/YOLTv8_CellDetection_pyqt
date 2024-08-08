import os
import cv2


def prefunc(img):  # 提升图像质量
    clahe = cv2.createCLAHE(clipLimit=8)
    imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    channelsYUV = cv2.split(imgYUV)
    channelsYUV_list = list(channelsYUV)
    channelsYUV_list[0] = clahe.apply(channelsYUV[0])
    # channelsYUV = tuple(channelsYUV_list)
    # channels = cv2.merge(channelsYUV)
    # result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
    # cv2.imwrite('res'+image, channelsYUV_list[0])

    return cv2.cvtColor(channelsYUV_list[0], cv2.COLOR_GRAY2BGR)


def compute_IOU(rec1, rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    x, y, w, h
    """
    loc1 = [rec1[0], rec1[1], rec1[0] + rec1[2], rec1[1] + rec1[3]]
    loc2 = [rec2[0], rec2[1], rec2[0] + rec2[2], rec2[1] + rec2[3]]
    left_column_max = max(loc1[0], loc2[0])
    right_column_min = min(loc1[2], loc2[2])
    up_row_max = max(loc1[1], loc2[1])
    down_row_min = min(loc1[3], loc2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (loc1[2] - loc1[0]) * (loc1[3] - loc1[1])
        S2 = (loc2[2] - loc2[0]) * (loc2[3] - loc2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (S1 + S2 - S_cross)


if __name__ == '__main__':

    # prefunc('1.jpg')
    # prefunc('2.jpg')
    dir = 'imagesa'
    res = 'images'
    for image in os.listdir(dir):
        cv2.imwrite(res + os.sep + image, prefunc(dir + os.sep + image))
