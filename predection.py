from ultralytics import YOLO


model = YOLO(r'weights/thick-9.6.pt')  # pretrained YOLOv8n model
source = r'D:\SCAN\24_09_07_02_45_14\trench0'
results = model(source, save=True, save_txt=True, save_conf=True, conf=0.25)