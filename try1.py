
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import yoltv8
from ultralytics import YOLO
import os
model = YOLO(r"C:\Users\nashi\Desktop\malariadetect\weights\classify_9.9.pt")

path = Path(r'E:\czc\6-result-8.30\PM\crop')

PF, PM, PO, POPV, PV = 0, 0, 0, 0, 0

conf_numbers = 0
file_numbers = 0
patient_status = '阴性'

# 创建保存图像的文件夹
save_folder = "output"
os.makedirs(save_folder, exist_ok=True)

def check_path_not_empty(path):
    if not os.path.exists(path):
        print(f"路径 {path} 不存在。")
        return False
    if not os.listdir(path):
        print(f"路径 {path} 为空。")
        return False
    return True


if check_path_not_empty(path):
    results = model(path, save=True, )
    for result in results:
        probs = result.probs.top1
        # print(result.probs.top5conf)
        name = results[0].names[int(probs)]
        n = 0
        for top in result.probs.top5:
            result_name = result.names[int(top)]
            conf = result.probs.top5conf[n]
            n = n + 1
            # print(f"{result_name} 的置信度为 {conf:.2f}")
            cv2.putText(result.orig_img, f"{result_name} {conf:.2f}", (10, 20 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
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

    Total_malaria = + PF + PM + PO + PV + POPV
    percentages = {
        'PF': (PF / Total_malaria) * 100,
        'PM': ((PM) / Total_malaria) * 100,
        'PO': ((PO) / Total_malaria) * 100,
        'PV': ((PV) / Total_malaria) * 100,
        'POPV': ((POPV) / Total_malaria) * 100
    }
    max_key = max(percentages, key=percentages.get)

    if file_numbers >= 5 or conf_numbers >= 1:
        patient_status = '阳性'

    print(f"当前病人为：{patient_status}")
    for k, v in percentages.items():
        print(f"{k}的概率是 {v:.2f}")
    print('------------------------')
    print(f"判断当前虫型为 {max_key}")
    print(Total_malaria)
else:
    print(f"当前病人为：{patient_status}")