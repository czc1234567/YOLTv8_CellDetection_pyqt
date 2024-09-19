import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import yoltv8
from ultralytics import YOLO
import os
import shutil

class malaria_inference():
    def __init__(self, modelpath, conf, iou):
        self.modelpath = modelpath
        self.conf = conf
        self.iou = iou
        self.predictor = yoltv8.malaria_predict.malaria_inference(self.modelpath, conf=self.conf, iou=self.iou)

    def infer(self, image_path):
        prediction_result = self.predictor.predict(image_path)
        conf_number, file_number = self.crop(prediction_result, image_path)
        return prediction_result, conf_number, file_number

    def crop(self, prediction_result, image_path):
        save_dir = Path('dataset/thin/crop')
        im_ext = '.jpg'
        os.makedirs(save_dir, exist_ok=True)

        im = cv2.imread(image_path)
        image_name = os.path.basename(image_path).split('.')[0]
        if im is None or im.ndim < 2:
            print("Failed to load image or image is not a 2D or 3D array.")
            return 0, 0
        conf_number = 0
        file_number = 0
        try:
            conf_number = sum(1 for box in prediction_result.boxes if int(box.cls) != 0 and box.conf >= 0.75)
        except AttributeError as e:
            print(f"Error: {e}. 'boxes' attribute not found in prediction_result.")
            return conf_number, file_number

        # conf_number = sum(1 for conf in prediction_result.boxes.conf if conf >= 0.85)
        for box in prediction_result.boxes:
            try:
                if int(box.cls) == 0:
                    continue
                xyxy = box.xyxy[0]
                if len(xyxy) < 4:
                    print("Boundary box coordinates are invalid.")
                    continue

                x1, y1, x2, y2 = map(int, xyxy[:4])
                width, height = x2 - x1, y2 - y1
                max_side = max(width, height)
                pad_x = max(0, (max_side - width) // 2)
                pad_y = max(0, (max_side - height) // 2)

                new_x1 = max(0, x1 - pad_x)
                new_y1 = max(0, y1 - pad_y)
                new_x2 = min(im.shape[1], x2 + pad_x)
                new_y2 = min(im.shape[0], y2 + pad_y)

                crop = im[new_y1:new_y2, new_x1:new_x2]
                if crop.size > 0:
                    if width < 2 * height and height < 2 * width and width*height<120000 and width*height>25600:
                        print(width, height)
                        file_path = os.path.join(save_dir, image_name + str(file_number) + im_ext)
                        cv2.imwrite(str(file_path), crop)
                        print(f"图像已保存：{file_path}")
                        file_number += 1
                else:
                    print("裁剪区域无效，跳过保存。")
            except Exception as e:
                print(f"Error processing box: {e}")

        return conf_number, file_number


if __name__ == '__main__':
    modelpath = r'C:\Users\nashi\Desktop\malariadetect\weights\thin_9.4_detection.pt'
    pre = malaria_inference(modelpath, conf=0.50, iou=0.1)
    images_path = r'D:\SCAN\24_09_09_01_47_36\trench0'

    PF, PM, PO, POPV, PV= 0, 0, 0, 0, 0

    conf_numbers = 0
    file_numbers = 0
    patient_status = '阴性'
    # 创建保存图像的文件夹
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

    for filename in os.listdir(images_path):
        file_path = os.path.join(images_path, filename)
        _, conf_number, file_number = pre.infer(file_path)
        conf_numbers += conf_number
        file_numbers += file_number

    model = YOLO(r"C:\Users\nashi\Desktop\malariadetect\weights\classify_9.9.pt")

    path = Path('dataset/thin/crop')
    save_path = Path('dataset/thin/crop_result')

    def check_path_not_empty(path):
        if not os.path.exists(path):
            print(f"路径 {path} 不存在。")
            return False
        if not os.listdir(path):
            print(f"路径 {path} 为空。")
            return False
        return True


    if check_path_not_empty(path):
        results = model(path,save=True,)
        for result in results:
            probs = result.probs.top1
            # print(result.probs.top5conf)
            name = results[0].names[int(probs)]
            n = 0
            for top in result.probs.top5:
                result_name = result.names[int(top)]
                conf = result.probs.top5conf[n]
                n = n + 1
                print(f"{result_name} 的置信度为 {conf:.2f}")
                cv2.putText(result.orig_img, f"{result_name} {conf:.2f}", (10, 20 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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


        Total_malaria = + PF + PM + PO  + PV +POPV
        percentages = {
            'PF': (PF / Total_malaria) * 100,
            'PM': ((PM ) / Total_malaria) * 100,
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
