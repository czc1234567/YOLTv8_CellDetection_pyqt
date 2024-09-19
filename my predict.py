import os
import sys
import argparse
from ultralytics import YOLO
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))
# from utils.slice_images import slice_image
# from utils.convert_coordinates import convert_coordinates
# from utils.draw_pred_on_onr_img import draw_predictions_on_image
import cv2
import time
import numpy as np
from ultralytics.engine.results import Results


class malaria_inference:


    def __init__(self, modelpath,conf, iou):
        self.modelpath = modelpath
        # self.model = YOLO(modelpath)

        self.conf = conf
        self.iou = iou

    def cvimread(self,path):
        """读取中文路径图片."""
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

    def convert_coordinates(self,
                            # 输入参数列表，包括检测结果的 TXT 文件路径、输出目录、IoU 阈值等
                            txt_label_path, output_file_dir, iou_threshold, confidence_threshold, area_weight, slice_sep
                            ):
        # txt_file_path: 存放 YOLOv8 小图检测结果的 TXT 文件的上级路径
        # output_file_path: 变换后的结果存放的 TXT 文件路径
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
            print(f"Created folder {output_file_dir}")
        output_lines = dict()  # 存储转换后的结果

        # orgimg_w = 0
        # orgimg_h = 0
        # 遍历文件夹中的每个 TXT 文件
        for root, dirs, files in os.walk(txt_label_path):
            for index, filename in enumerate(files):
                # 如果文件以 .txt 结尾，表示它是一个检测结果文件
                if filename.endswith(".txt"):
                    # 获取文件的完整路径
                    filepath = os.path.join(root, filename)

                    # 解析文件名中的信息
                    slice_info = filename.split(".")[0].split(slice_sep)
                    y0 = int(slice_info[-6])
                    x0 = int(slice_info[-5])
                    sliceHeight = int(slice_info[-4])
                    sliceWidth = int(slice_info[-3])
                    orgimg_w = int(slice_info[-2])
                    orgimg_h = int(slice_info[-1])

                    exclude_imgname_char = slice_sep + str(y0) + slice_sep + str(x0) + slice_sep + str(sliceHeight) + \
                                           slice_sep + str(sliceWidth) + slice_sep + str(orgimg_w) + slice_sep + str(
                        orgimg_h)
                    exclude_imgname_index = filename.split(".")[0].index(exclude_imgname_char)
                    imgname = filename.split(".")[0][:exclude_imgname_index]

                    # 读取小图检测结果的坐标信息
                    with open(filepath, "r") as f:
                        lines = f.readlines()

                    # 将边界框坐标转换到原图的坐标空间，并将结果存储到列表中
                    converted_lines = []
                    for line in lines:
                        class_label, x, y, w, h, conf = line.strip().split(" ")
                        x = float(x) * sliceWidth
                        y = float(y) * sliceHeight
                        w = float(w) * sliceWidth
                        h = float(h) * sliceHeight

                        # # 未归一化，因为后文需要
                        # x_in_original = (float(x) + x0) / orgimg_w
                        # y_in_original = (float(y) + y0) / orgimg_h
                        # w_in_original = float(w) / orgimg_w
                        # h_in_original = float(h) / orgimg_h
                        # converted_line = f"{class_label} {x_in_original} {y_in_original} {w_in_original} {h_in_original} {conf}\n"
                        '''
                        x0 和 y0 代表的是图像切片的左上角坐标
                        举个例子，如果原始图像的大小是 3840x2160 像素，你从中切出了一个 640x640 像素的切片，
                        而 (x0, y0) 是 (1000, 1000)，这意味着切片的左上角位于原始图像的坐标 (1000, 1000)。
                        如果在这个切片中检测到一个对象，其坐标 (x, y) 是 (320, 240)，那么在原始图像中，该对象的坐标将是 (1320, 1240)。
                        '''
                        x_in_original = float(x) + x0
                        y_in_original = float(y) + y0
                        w_in_original = float(w)
                        h_in_original = float(h)
                        converted_line = [
                            int(class_label),
                            x_in_original,
                            y_in_original,
                            w_in_original,
                            h_in_original,
                            float(conf),
                            orgimg_w,
                            orgimg_h
                        ]

                        converted_lines.append(converted_line)
                        # print(converted_lines)

                    # 将转换后的结果添加到输出列表中
                    if imgname not in output_lines.keys():
                        output_lines[imgname] = converted_lines
                    else:
                        output_lines[imgname].extend(converted_lines)

        # print(f'orgimg_w-------{orgimg_w}')
        # print(f'orgimg_h-------{orgimg_h}')

        outputs_file_path_list = []
        for key, value in output_lines.items():
            nms_output_lines = self.apply_nms(
                value,
                iou_threshold,
                confidence_threshold,
                area_weight,
            )

            # 将转换后的结果写入输出文件
            output_file_path = os.path.join(output_file_dir, f"{key}.txt")
            if os.path.exists(output_file_path):
                # import shutil
                import logging
                os.remove(output_file_path)
                logging.warning(
                    f"completed predict txt results of image—{key} have been existed! The original content will be overwritten!")

            with open(output_file_path, "w") as f:
                f.writelines(nms_output_lines)
            print(f"completed predict txt results of image—{key} is saved at: {output_file_path}")
            outputs_file_path_list.append(output_file_path)

        return outputs_file_path_list

    def slice_image(self,
                    image_path,
                    out_dir_all_images,
                    sliceHeight=640,
                    sliceWidth=640,
                    overlap=0.1,
                    slice_sep="_",
                    overwrite=False,
                    out_ext=".png",
                    ):
        if len(out_ext) == 0:
            im_ext = "." + image_path.split(".")[-1]
        else:
            im_ext = out_ext

        # 清空输出路径中的所有文件
        if os.path.exists(out_dir_all_images):
            for filename in os.listdir(out_dir_all_images):
                file_path = os.path.join(out_dir_all_images, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 删除文件
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 删除目录
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        t0 = time.time()
        image = self.cvimread(image_path)  # , as_grey=False).astype(np.uint8)  # [::-1]
        print("image.shape:", image.shape)

        image_name = os.path.basename(image_path).split('.')[0]
        win_h, win_w = image.shape[:2]

        dx = int((1.0 - overlap) * sliceWidth)
        dy = int((1.0 - overlap) * sliceHeight)

        out_dir_image = os.path.join(out_dir_all_images)

        n_ims = 0
        for y0 in range(0, image.shape[0], dy):
            for x0 in range(0, image.shape[1], dx):
                n_ims += 1

                if (n_ims % 100) == 0:
                    print(n_ims)

                # make sure we don't have a tiny image on the edge
                if y0 + sliceHeight > image.shape[0]:
                    y = image.shape[0] - sliceHeight
                else:
                    y = y0
                if x0 + sliceWidth > image.shape[1]:
                    x = image.shape[1] - sliceWidth
                else:
                    x = x0

                # extract image
                window_c = image[y: y + sliceHeight, x: x + sliceWidth]
                outpath = os.path.join(
                    out_dir_image,
                    image_name
                    + slice_sep
                    + str(y)
                    + "_"
                    + str(x)
                    + "_"
                    + str(sliceHeight)
                    + "_"
                    + str(sliceWidth)
                    + "_"
                    + str(win_w)
                    + "_"
                    + str(win_h)
                    + im_ext,
                )
                if not os.path.exists(outpath):
                    cv2.imwrite(outpath, window_c)
                elif overwrite:
                    cv2.imwrite(outpath, window_c)
                else:
                    print("outpath {} exists, skipping".format(outpath))

        print("Num slices:", n_ims, "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
        print("Time to slice", image_path, time.time() - t0, "seconds")
        print(
            f"cliped results of {os.path.basename(image_path)} is saved at: {out_dir_image}"
        )
        return out_dir_image

    def calculate_area(self, box):
        """
        计算边界框的面积
        box的格式：[xmin, ymin, xmax, ymax]
        """
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        return area

    def calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU（Intersection over Union）
        box1和box2的格式：[xmin, ymin, xmax, ymax]
        """
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # 计算交集的坐标
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)

        if x_right < x_left or y_bottom < y_top:
            # 两个边界框没有交集
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # 计算并集的面积
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area
        return iou

    def nms_per_class(self,
                      boxes, scores, classes, iou_threshold, confidence_threshold, area_weight
                      ):
        """
        使用NMS对不同类别的边界框进行后处理
        boxes: 边界框列表，每个边界框的格式为 [xmin, ymin, xmax, ymax]
        scores: 每个边界框的置信度得分列表
        classes: 每个边界框的类别列表
        threshold: 重叠度阈值，高于该阈值的边界框将被抑制
        """
        # 过滤置信度低于阈值的边界框
        filtered_indices = np.where(np.array(scores) >= confidence_threshold)[0]
        boxes = [boxes[i] for i in filtered_indices]
        scores = [scores[i] for i in filtered_indices]
        classes = [classes[i] for i in filtered_indices]

        # 将边界框、置信度、类别转换为NumPy数组
        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array(classes)
        areas = np.array([self.calculate_area(box) for box in boxes])

        # 初始化空列表来存储保留的边界框索引
        keep_indices = []

        # 获取所有唯一的类别标签
        unique_classes = np.unique(classes)

        for cls in unique_classes:
            # 获取属于当前类别的边界框索引
            '''
            classes = np.array([1, 2, 3, 2, 1, 4])
            如果我们想要找出数组中所有类别为 2 的索引，我们可以这样做：
            cls = 2
            cls_indices = np.where(classes == cls)[0]
            print(cls_indices)  # 输出: [1 3]
            在这个例子中，cls_indices 将会是 [1, 3]，因为 classes 数组中索引为 1 和 3 的元素等于 2。
            注意事项：
    np.where(classes == cls)[0] 中的 [0] 是因为 np.where 返回的是一个元组，其中第一个元素包含了满足条件的索引。使用 [0] 来获取这个索引数组。
    如果 cls 在 classes 中不存在，cls_indices 将会是一个空数组。
    这种方法适用于一维数组。如果 classes 是多维的，你可能需要使用不同的索引方式或遍历逻辑。
    np.where 是一个非常有用的函数，可以快速找出满足特定条件的元素位置，它在数据处理和科学计算中非常有用。
            '''
            cls_indices = np.where(classes == cls)[0]

            # 根据当前类别的置信度得分和面积对边界框进行排序
            sorted_indices = np.lexsort(
                (scores[cls_indices], areas[cls_indices] * area_weight)
            )[::-1]
            # sorted_indices = np.argsort(areas[cls_indices])[::-1]
            cls_indices = cls_indices[sorted_indices]
            while len(cls_indices) > 0:
                # 选择当前得分最高的边界框
                current_index = cls_indices[0]
                current_box = boxes[current_index]
                keep_indices.append(filtered_indices[current_index])

                # 计算当前边界框与其他边界框的IoU
                other_indices = cls_indices[1:]
                ious = np.array(
                    [self.calculate_iou(current_box, boxes[i]) for i in other_indices]
                )

                # 找到重叠度低于阈值的边界框索引
                low_iou_indices = np.where(ious < iou_threshold)[0]

                # 更新剩余边界框索引
                cls_indices = cls_indices[1:][low_iou_indices]

        return keep_indices

    def apply_nms(self,
                  outputs, iou_threshold, confidence_threshold, area_weight
                  ):
        # 将边界框列表转换为NumPy数组
        outputs = np.array(outputs)

        boxes = []
        scores = []
        class_ids = []
        for out in outputs:
            x = out[1]
            y = out[2]
            w = out[3]
            h = out[4]
            score = out[5]
            class_id = int(out[0])
            orgimg_w = int(out[6])
            orgimg_h = int(out[7])

            # 计算边界
            left = float(x - w / 2)
            top = float(y - h / 2)
            right = float(x + w / 2)
            bottom = float(y + h / 2)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(score)
            boxes.append([left, top, right, bottom])

        # 应用NMS
        # indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
        indices = self.nms_per_class(
            boxes=boxes,
            scores=scores,
            classes=class_ids,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            area_weight=area_weight,
        )

        # 选择通过NMS过滤后的边界框
        nms_out_lines = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            x = float(box[0] + (box[2] - box[0]) / 2) / orgimg_w
            y = float(box[1] + (box[3] - box[1]) / 2) / orgimg_h
            w = float(box[2] - box[0]) / orgimg_w
            h = float(box[3] - box[1]) / orgimg_h
            nms_out_line = f"{class_id} {x} {y} {w} {h} {score}\n"
            nms_out_lines.append(nms_out_line)
        return nms_out_lines

    def draw_predictions_on_image(self,
                                  image_path, results_file_path, class_labels, class_names, completed_output_path
                                  ):
        # 确保类别标签和类别名称数量一致
        assert len(class_labels) == len(
            class_names
        ), "Number of class labels should match the number of class names."

        # 定义每个类别对应的颜色
        colors = [
            (255, 0, 0),  # head: 红色
            (0, 255, 0),  # boxholder: 绿色
            (0, 0, 255),  # greendevice: 蓝色
            (255, 255, 0),  # baseholer: 青色
            (0, 255, 255),  # circledevice: 黄色
            (255, 0, 255),  # alldrop: 品红色
        ]

        # 定义类别标签到类别名称的映射
        label_map = dict(zip(class_labels, class_names))

        # 定义每个类别对应的颜色
        color_map = dict(zip(class_labels, colors))

        # 读取原图像
        image = cv2.imread(image_path)
        a = []
        boxes_lists =[]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 读取存放变换结果的文本文件
        with open(results_file_path, 'r') as file:
            lines = file.readlines()

        # 遍历每行结果
        for line in lines:
            line = line.strip().split(' ')
            class_label, x, y, w, h, conf = map(float, line)

            # 计算边界框的坐标
            image_height, image_width, _ = image.shape
            abs_x = int(abs(x * image_width))
            abs_y = int(abs(y * image_height))
            abs_w = int(abs(w * image_width))
            abs_h = int(abs(h * image_height))
            x_min = abs_x - abs_w // 2
            y_min = abs_y - abs_h // 2
            x_max = abs_x + abs_w // 2
            y_max = abs_y + abs_h // 2
            boxes_list = [x_min, y_min, x_max, y_max,conf,class_label]
            boxes_lists.append(boxes_list)
            # 获取类别名称和颜色
            class_name = label_map.get(int(class_label), "Unknown")
            color = color_map[int(class_label)]


            # 绘制边界框和类别标签
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(
                image,
                f"{class_name}: {conf:.2f}",
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        # images_path = r'C:\Users\xuzhe\Desktop\First_task\v8_High_resolution\input\_0_1.png'

        image2 = cv2.imread(image_path)
        if boxes_lists ==[]:
            boxes_lists = np.empty((0, 6))
        boxes_array = np.array(boxes_lists)
        a = Results(orig_img=image2, path=image_path, names=class_names, boxes=boxes_array)
        # 保存绘制结果
        filename = os.path.basename(image_path)
        if not os.path.exists(completed_output_path):
            os.makedirs(completed_output_path)

        output_image_path = os.path.join(completed_output_path, filename)
        if os.path.exists(output_image_path):
            # import shutil
            import logging
            os.remove(output_image_path)
            logging.warning(
                f"completed predict visual result of image-{filename} have been existed! The original content will be overwritten!")

        cv2.imwrite(output_image_path, image)
        print(f"completed predict visual result of image-{filename} is saved at: {output_image_path}")



        return a

    def predict(self, images_path):
        outdir_slice_ims = os.path.join(PROJECT_ROOT, 'dataset', 'thin', 'slice_images')
        slice_sep = '_'
        output_file_dir = os.path.join(PROJECT_ROOT, 'results', 'completed_txt')
        iou_threshold = 0.1
        confidence_threshold = 0.7
        area_weight = 10
        # draw arg
        # class_labels = [0, 1,2,3,4]
        # class_names = [
        #     "WBC",
        #     "malaria",
        # ]
        # class_names = [
        #     "WBC",
        #     'PO',
        #     'PF',
        #     'PV',
        #     'PM',
        # ]
        IMAGE_EXTENSIONS = ['.png', '.jpg']
        completed_output_path = os.path.join(PROJECT_ROOT, 'results', 'completed_predict')
        def is_image_file(path):
            """判断文件是否是图片"""
            if not isinstance(path, str):
                return False
            return any(path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
        self.slice_image(
            images_path,
            outdir_slice_ims,
            sliceHeight=640,
            sliceWidth=640,
            overlap=0.3,
            slice_sep='_',
            overwrite=False,
            out_ext='.png',
        )

        yolov8_predict_results_path = os.path.join(PROJECT_ROOT, 'results', 'yolov8_detect', 'predict')

        if os.path.exists(yolov8_predict_results_path):
            import shutil
            import logging
            shutil.rmtree(yolov8_predict_results_path)
            logging.warning(
                f"detect predict path: {yolov8_predict_results_path} is existed! The original content will be overwritten!")
        self.modelpath
        model = YOLO(self.modelpath)  # pretrained YOLOv8n model
        source = os.path.join(outdir_slice_ims)
        results = []
        results = model(source, save=True, save_txt=True, save_conf=True, conf=0.65, project='results/yolov8_detect')
        original_dict = results[0].names
        class_labels = []
        class_names = []
        # 遍历字典，将键赋值给 class_label，将值赋值给 class_names
        for num, name in original_dict.items():
            class_labels.append(num)
            class_names.append(name)


        txt_label_path = os.path.join(yolov8_predict_results_path, 'labels')

        txt_regress_path_list = self.convert_coordinates(
            txt_label_path=txt_label_path,
            output_file_dir=os.path.join(output_file_dir),
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            area_weight=area_weight,
            slice_sep=slice_sep
        )



        if txt_regress_path_list ==[]:
            image2 = cv2.imread(images_path)
            boxes_lists = np.empty((0, 6))
            boxes_array = np.array(boxes_lists)
            results = Results(orig_img=image2, path=images_path, names=class_names, boxes=boxes_array)


        elif os.path.isfile(images_path) and is_image_file(images_path):
            for txt_regress_path in txt_regress_path_list:
                # image_name = os.path.basename(txt_regress_path).split('.')[0]
                #
                # image_path = os.path.join(image_path, image_name + '.png')
                results = self.draw_predictions_on_image(
                    image_path=images_path,
                    results_file_path=txt_regress_path,
                    class_labels=class_labels,
                    class_names=class_names,
                    completed_output_path=os.path.join(completed_output_path),
                )

        elif os.path.isdir(images_path):
            for txt_regress_path in txt_regress_path_list:
                image_name = os.path.basename(txt_regress_path).split('.')[0]

                im_path = os.path.join(images_path, image_name + '.png')

                results = self.draw_predictions_on_image(
                    image_path=im_path,
                    results_file_path=txt_regress_path,
                    class_labels=class_labels,
                    class_names=class_names,
                    completed_output_path=os.path.join(completed_output_path),
                )

        return results

if __name__ == '__main__':
    images_path = r'C:\Users\xuzhe\Desktop\First_task\malariadetect\dataset\thin\init_images\张三-daiwen-0016.jpg'
    modelpath = r'C:\Users\xuzhe\Desktop\First_task\malariadetect\weights\thin-detection_8.6.pt'
    pre = malaria_inference(modelpath,0.5, 0.4)
    pres = pre.predict(images_path)
    print(pres.boxes.xywh)
