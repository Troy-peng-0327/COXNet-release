import os
import pickle
import json
import cv2
import numpy as np

# Load the results from the pkl file
pkl_file_path = 'work_dir/rgbtdroneperson/vis_results/visulization/ours/out.pkl'
with open(pkl_file_path, 'rb') as f:
    results = pickle.load(f)

# Load the ground truth JSON file
json_file_path = '/data3/pengpeiran/datasets/RGBTDronePerson/vis.json'
with open(json_file_path, 'r') as f:
    gt_data = json.load(f)

# Define colors for bounding boxes
correct_color = (0, 255, 0)  # Green for correct detections
incorrect_color = (0, 0, 255)  # Red for incorrect detections
missed_color = (0, 255, 255)  # Yellow for missed detections

# Function to compute IoU (Intersection over Union)
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# Iterate through the images and draw bounding boxes
for i, (image_meta, result) in enumerate(zip(gt_data['images'], results)):
    filename = image_meta['file_name']
    image_id = image_meta['id']
    
    # Load the image
    image_path = os.path.join('/data3/pengpeiran/datasets/RGBTDronePerson/val/thermal', filename)
    image = cv2.imread(image_path)
    
    # Get the ground truth bboxes for the current image
    gt_bboxes = [ann['bbox'] for ann in gt_data['annotations'] if ann['image_id'] == image_id]
    gt_bboxes = [ [x, y, x+w, y+h] for x, y, w, h in gt_bboxes ]  # Convert to [x1, y1, x2, y2]

    # Get the predicted bboxes from the results
    pred_bboxes = result[0]
    pred_bboxes = [bbox[:4] for bbox in pred_bboxes]  # Ignore scores
    
    # Create a list to track which gt bboxes have been matched
    matched_gts = [False] * len(gt_bboxes)
    
    # Draw the predicted bboxes
    for pred_bbox in pred_bboxes:
        match_found = False
        for j, gt_bbox in enumerate(gt_bboxes):
            iou = compute_iou(pred_bbox, gt_bbox)
            if iou >= 0.1:
                match_found = True
                matched_gts[j] = True
                cv2.rectangle(image, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), correct_color, 2)
                break
        if not match_found:
            cv2.rectangle(image, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), incorrect_color, 2)
    
    # Draw the missed ground truth bboxes
    for k, gt_bbox in enumerate(gt_bboxes):
        if not matched_gts[k]:
            cv2.rectangle(image, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), missed_color, 2)
    
    # Save the output image with bounding boxes
    output_path = os.path.join('work_dir/rgbtdroneperson/vis_results/visulization_fp', filename)
    cv2.imwrite(output_path, image)