import glob
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.transform = transform
        self.data = data

    def __getitem__(self, item):
        data = self.transform(self.data[item])
        return data

    def __len__(self):
        return len(self.data)


def makeData(imgList, imgsz=320):
    transform = transforms.Compose([
        transforms.Resize([imgsz, imgsz]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    imgs = [transform(Image.fromarray(img)) for img in imgList]
    data = torch.stack(imgs)
    return data
