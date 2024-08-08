import time

from ultralytics import YOLO
import os
import torch
import cv2
from core.myAlgo import cvimread, cvimwrite
from core.se_resnet import se_resnet
import torchvision.transforms as transforms
from PIL import Image

"""
2023.5.10 liuyulin

readme:
classify inference results formats as follow:
boxes: None
keypoints: None
keys: ['probs']
masks: None
names: {0: 'thick', 1: 'thin'}
orig_img: array([[[161, 129, 100], #img dataset
orig_shape: (1944, 2592)
path: 'C:\\Users\\alfeak\\Desktop\\deepin\\yolov8\\datasets\\smear2\\test\\fjx0073.jpg'
probs: tensor([4.6979e-04, 9.9953e-01], device='cuda:0')
speed: {'preprocess': 0.0, 'inference': 2.991914749145508, 'postprocess': 0.0}]

detection inference results formats as follow:
boxes.xyxy  # box with xyxy format, (N, 4)
boxes.xywh  # box with xywh format, (N, 4)
boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
boxes.xywhn  # box with xywh format but normalized, (N, 4)
boxes.conf  # confidence score, (N, 1)
boxes.cls  # cls, (N, 1)
boxes.dataset  # raw bboxes tensor, (N, 6) or boxes.boxes
speed and other conmmon information same as classify
"""


class inferenceT:
    def __init__(self, modelpath):
        self.model = YOLO(modelpath)

    def infer(self, imgpath, device='cpu'):
        results = self.model.predict(source=imgpath, imgsz=960, conf=0.20)
        return results


class inferenceC:
    def __init__(self, modelpath):
        self.modelpath = modelpath
        self.model = YOLO(modelpath)
        self.res = None
        self.imgres = None
        self.infertime = None
        self.cls = None

    def infer(self, imgpath, device='cpu'):
        results = self.model(imgpath, device=device, verbose=False)[0]
        self.imgres = results.orig_img
        self.infertime = str(
            float(results.speed['preprocess']) + float(results.speed['inference'])) + 'ms'
        self.cls = results.names
        self.res = self.cls[torch.topk(results.probs, 1)[1].item()], torch.topk(
            results.probs, 1)[0].item()
        return self.res

    def save(self, imgpath):
        self.imgres = cv2.resize(self.imgres, (640, 640))
        cv2.putText(self.imgres, str(self.res[0]) + ' ' + str(format(self.res[1], '.3f')),
                    (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(imgpath, self.imgres)

    def time(self):
        return self.infertime


class inferenceResnet:
    def __init__(self, modelpath):
        self.modelpath = modelpath
        self.model = se_resnet(224, 2)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize([320, 320]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.model.load_state_dict(torch.load(modelpath, map_location='cpu'))
        self.classes = ['thin', 'thick']

    def infer(self, img):
        img = Image.fromarray(img)
        with torch.no_grad():
            output = self.model(self.transform(img).unsqueeze(0))
            index, predicted = torch.max(output, 1)
        return [self.classes[predicted], index.item()]


class inferenceD:
    def __init__(self, modelpath, conf, iou):
        self.modelpath = modelpath
        self.model = YOLO(modelpath)
        self.res = []
        self.imgres = None
        self.conf = conf
        self.iou = iou
        self.infertime = None
        self.cls = None

    def infer(self, imgpath, device='cpu'):


        self.results = self.model(
            imgpath, device=device, conf=self.conf, iou=self.iou, verbose=False)[0]


        # for result in self.results:
        #     self.res.append(result.boxes)

        return self.results

    def save(self, imgpath):
        self.imgres = self.results.plot()
        cv2.imwrite(imgpath, self.imgres)

    def time(self):
        cost = float(self.results.speed['preprocess']) + \
               float(self.results.speed['inference'])
        self.infertime = f'{cost} ms'
        return self.infertime

    def crop(self, cat):
        for i, xyxy in enumerate(self.results.boxes.xyxy.tolist()):
            x1, y1, x2, y2 = map(int, xyxy)
            path = os.path.join(cat, str(i).zfill(4) + '.jpg')
            cvimwrite(path, self.results.orig_img[y1:y2, x1:x2])


# using examples
if __name__ == '__main__':
    inputDir = r'D:\scanPath\24_05_24_03_30_16'
    infer = inferenceD(r'C:\code\malariadetect\weights\thin-P00.pt', 0.2, 0.5)

    # res = {}
    # starttime = time.time()
    # for name in os.listdir(inputDir):
    #     path = os.path.join(inputDir, name)
    #     res[path] = {}
    #     bgr = cvimread(path)
    #     pred = infer.infer(bgr)
    #     for xywh, class_, conf in zip(pred.boxes.xywh, pred.boxes.cls, pred.boxes.conf):
    #         cx, cy, w, h = map(int, xywh)
    #         x = int(cx - w / 2)
    #         y = int(cy - h / 2)
    #         print("{}疟原虫,置信度{},path={}".format(class_, conf.item(), name))
    #         cell = bgr[y:y + h, x:x + w].copy()
    #         cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255))
    #     save = r'C:\dataset\Test_result'
    #     cv2.imwrite(os.path.join(save, name), bgr)
    # print("time:", time.time() - starttime)
    # print(len(os.listdir(inputDir)))

    thicknessClassifier = inferenceResnet(r'C:\code\malariadetect\weights\Classifier.pth')

    for name in os.listdir(inputDir):
        path = os.path.join(inputDir, name)
        bgr = cvimread(path)
        res = thicknessClassifier.infer(bgr)
        print(res)

