import pickle
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_iou(box1, box2):
    """计算两个矩形框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union != 0 else 0

def visualize_results(image_path, gt_boxes, pred_boxes, iou_threshold=0.1, output_dir=None):
    """
    可视化检测结果
    Args:
        image_path: 图像路径
        gt_boxes: 真值框 (x1, y1, x2, y2)
        pred_boxes: 预测框 (x1, y1, x2, y2, score)
        iou_threshold: 判断正确检测的IoU阈值
    """
    img = cv2.imread(image_path)

    # 遍历真值框
    for gt in gt_boxes:
        matched = False
        for pred in pred_boxes:
            iou = calculate_iou(gt, pred[:4])
            if iou >= iou_threshold:
                # 正确检测
                cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2)
                matched = True
        if not matched:
            # 漏检
            cv2.rectangle(img, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 165, 255), 2)

    # 遍历预测框
    for pred in pred_boxes:
        matched = False
        for gt in gt_boxes:
            iou = calculate_iou(gt, pred[:4])
            if iou >= iou_threshold:
                matched = True
                break
        if not matched:
            # 错误检测
            cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255), 2)

    # # 显示图像
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    # 保存图像
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    cv2.imwrite(output_dir, img)

# 加载真值数据
with open('/data3/pengpeiran/datasets/RGBTDronePerson/vis.json', 'r') as f:
    gt_data = json.load(f)

# 加载预测数据
with open('work_dir/vis_results/rgbtdroneperson/coxnet_star/out.pkl', 'rb') as f:
    pred_data = pickle.load(f)

output_dir = "work_dir/vis_results/rgbtdroneperson/coxnet_star/"

# 构建真值框索引
gt_boxes_dict = {}
for annotation in gt_data["annotations"]:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    x1, y1, width, height = bbox
    gt_box = [x1, y1, x1 + width, y1 + height]
    if image_id not in gt_boxes_dict:
        gt_boxes_dict[image_id] = []
    gt_boxes_dict[image_id].append(gt_box)

# 遍历图像进行可视化
for image_info in gt_data["images"]:
    image_id = image_info["id"]
    image_name = image_info["file_name"]
    image_path = f"/data3/pengpeiran/datasets/RGBTDronePerson/val/visible/{image_name}"  # 替换为实际图像路径
    gt_boxes = gt_boxes_dict.get(image_id, [])
    pred_boxes = pred_data.get(image_name, {}).get('pred_boxes', [])

    # 可视化
    visualize_results(image_path, gt_boxes, pred_boxes, iou_threshold=0.1, output_dir=output_dir)
