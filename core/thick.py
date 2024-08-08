import os

from ultralytics import YOLO



def detect(pic_path):

    # 模型权重文件
    weight = r"D:\code\thick\weights\best.pt"
    model = YOLO(weight)
    results = model.predict(source=pic_path, imgsz=960, conf=0.20)  # treat predict as a Python generator
    # for box in results[0].boxes:
    #     # result = []
    #     # cls = int(boxes.cls.numpy()[0])
    #     # x = round(boxes.xywh.numpy()[0][0], 2)
    #     # y = round(boxes.xywh.numpy()[0][1], 2)
    #     # w = round(boxes.xywh.numpy()[0][2], 2)
    #     # h = round(boxes.xywh.numpy()[0][3], 2)
    #     # conf = round(boxes.conf.numpy()[0], 2)
    #
    #     # print(cls, x, y, w, h, conf)
    #     aa = box.cls, box.xywh, box.conf
    #     return aa
    print(results)
    # boxes.xywh
    # boxes.conf
    # boxes.cls



if __name__ == "__main__":

    # 源文件路径
    source = r"E:\NIH-NLM-ThickBloodSmearsPV\im"

    img_name_list = os.listdir(source)
    for img_name in img_name_list:

        print(img_name)

        if not img_name.endswith('.jpg'):
            continue

        pic_path = os.path.join(source, img_name)

        detect(pic_path)