# import os
# import json
# import csv
# from PIL import Image
# from tqdm import tqdm

# # 路径设置
# img_ori_path_v_train = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images_ori/train/rgb'
# img_ori_path_t_train = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images_ori/train/thermal'
# img_ori_path_v_val = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images_ori/val/rgb'
# img_ori_path_t_val = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images_ori/val/thermal'

# label_txt_path_train = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/labels/train'
# label_txt_path_val = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/labels/val'

# img_path_v_train = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images/train/rgb'
# img_path_t_train = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images/train/thermal'
# img_path_v_val = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images/val/rgb'
# img_path_t_val = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images/val/thermal'

# json_train_path = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/train_rgb.json'
# json_val_path = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/val_rgb.json'

# # 创建目标目录
# os.makedirs(img_path_v_train, exist_ok=True)
# os.makedirs(img_path_t_train, exist_ok=True)
# os.makedirs(img_path_v_val, exist_ok=True)
# os.makedirs(img_path_t_val, exist_ok=True)

# def resize_and_create_coco_json(img_ori_v_dir, img_ori_t_dir, label_dir, img_v_dir, img_t_dir, output_json):
#     coco_output = {
#         "images": [],
#         "annotations": [],
#         "categories": [
#             {"id": 1, "name": "person", "supercategory": "none"}
#         ]
#     }
    
#     annotation_id = 1
#     image_id = 1

#     # 获取所有标签文件列表
#     label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

#     # 使用tqdm显示进度条
#     for label_file in tqdm(label_files, desc="Processing Images and Labels"):
#         file_id = label_file.replace('.txt', '')  # 例如 flight2_frame12421
#         txt_path = os.path.join(label_dir, label_file)
#         img_v_path = os.path.join(img_ori_v_dir, f"{file_id}.jpg")
#         img_t_path = os.path.join(img_ori_t_dir, f"{file_id}.jpg")
        
#         # 检查图像是否存在
#         if not os.path.exists(img_v_path):
#             print(f"RGB Image not found: {img_v_path}")
#             continue
#         if not os.path.exists(img_t_path):
#             print(f"Thermal Image not found: {img_t_path}")
#             continue
        
#         # 读取thermal图像的宽度和高度（用于将RGB图像缩放到thermal图像尺寸）
#         with Image.open(img_t_path) as img_t:
#             t_width, t_height = img_t.size
        
#         # 修改RGB图像的尺寸以匹配thermal图像，并保存到新的目录
#         with Image.open(img_v_path) as img_v:
#             img_v_resized = img_v.resize((t_width, t_height))
#             img_v_new_path = os.path.join(img_v_dir, f"{file_id}.jpg")
#             img_v_resized.save(img_v_new_path)
        
#         # 复制thermal图像到新目录（路径保持不变）
#         img_t_new_path = os.path.join(img_t_dir, f"{file_id}.jpg")
#         if not os.path.exists(img_t_new_path):
#             os.symlink(img_t_path, img_t_new_path)  # 创建软链接
        
#         # 添加图像信息到COCO格式
#         image_info = {
#             "id": image_id,
#             "file_name": f"{file_id}.jpg",
#             "width": t_width,
#             "height": t_height
#         }
#         coco_output['images'].append(image_info)
        
#         # 读取标签文件内容并调整bbox
#         with open(txt_path, newline='') as txtfile:
#             reader = csv.reader(txtfile, delimiter='\t')
#             for row in reader:
#                 x1, y1, x2, y2, obj_type, occluded, bad = map(float, row[:7])
                
#                 # 计算宽度和高度
#                 bbox_width = x2 - x1
#                 bbox_height = y2 - y1
#                 area = bbox_width * bbox_height
                
#                 # 生成COCO格式的annotation
#                 annotation = {
#                     "id": annotation_id,
#                     "image_id": image_id,
#                     "category_id": 1,  # 类别ID，假设为"person"
#                     "bbox": [x1, y1, bbox_width, bbox_height],
#                     "area": area,
#                     "iscrowd": 0,  # 假设无crowd标签
#                     "occluded": int(occluded),  # 保留遮挡信息
#                     "bad": int(bad)  # 保留bad字段信息
#                 }
#                 coco_output['annotations'].append(annotation)
#                 annotation_id += 1
        
#         image_id += 1

#     # 保存到JSON文件
#     with open(output_json, 'w') as json_file:
#         json.dump(coco_output, json_file, indent=4)

#     print(f'COCO格式文件已生成: {output_json}')

# # 生成train和val的json文件
# resize_and_create_coco_json(img_ori_path_v_train, img_ori_path_t_train, label_txt_path_train, img_path_v_train, img_path_t_train, json_train_path)
# resize_and_create_coco_json(img_ori_path_v_val, img_ori_path_t_val, label_txt_path_val, img_path_v_val, img_path_t_val, json_val_path)

import os
import json
import csv
from PIL import Image
from tqdm import tqdm

# 路径设置
img_ori_path_v_train = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images_ori/train/rgb'
img_ori_path_t_train = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images_ori/train/thermal'
img_ori_path_v_val = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images_ori/val/rgb'
img_ori_path_t_val = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images_ori/val/thermal'

label_txt_path_train = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/labels/train'
label_txt_path_val = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/labels/val'

img_path_v_train = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images/train/rgb'
img_path_t_train = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images/train/thermal'
img_path_v_val = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images/val/rgb'
img_path_t_val = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/images/val/thermal'

json_train_path = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/train_rgb.json'
json_val_path = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/val_rgb.json'

def adjust_bbox_and_create_coco_json(img_ori_v_dir, img_new_v_dir, label_dir, output_json):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "none"}
        ]
    }
    
    annotation_id = 1
    image_id = 1

    # 获取所有标签文件列表
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    # 使用tqdm显示进度条
    for label_file in tqdm(label_files, desc="Processing Labels"):
        file_id = label_file.replace('.txt', '')  # 例如 flight2_frame12421
        txt_path = os.path.join(label_dir, label_file)
        img_ori_v_path = os.path.join(img_ori_v_dir, f"{file_id}.jpg")
        img_new_v_path = os.path.join(img_new_v_dir, f"{file_id}.jpg")
        
        # 读取原始和新RGB图像的宽度和高度
        with Image.open(img_ori_v_path) as img_ori_v:
            ori_width, ori_height = img_ori_v.size
        
        with Image.open(img_new_v_path) as img_new_v:
            new_width, new_height = img_new_v.size
        
        # 比例因子
        scale_x = new_width / ori_width
        scale_y = new_height / ori_height
        
        # 添加图像信息到COCO格式
        image_info = {
            "id": image_id,
            "file_name": f"{file_id}.jpg",
            "width": new_width,
            "height": new_height
        }
        coco_output['images'].append(image_info)
        
        # 读取标签文件内容并调整bbox
        with open(txt_path, newline='') as txtfile:
            reader = csv.reader(txtfile, delimiter='\t')
            for row in reader:
                x1, y1, x2, y2, obj_type, occluded, bad = map(float, row[:7])
                
                # 调整 bbox
                x1_new = x1 * scale_x
                y1_new = y1 * scale_y
                x2_new = x2 * scale_x
                y2_new = y2 * scale_y
                
                bbox_width = x2_new - x1_new
                bbox_height = y2_new - y1_new
                area = bbox_width * bbox_height
                
                # 生成COCO格式的annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # 类别ID，假设为"person"
                    "bbox": [x1_new, y1_new, bbox_width, bbox_height],
                    "area": area,
                    "iscrowd": 0,  # 假设无crowd标签
                    "occluded": int(occluded),  # 保留遮挡信息
                    "bad": int(bad)  # 保留bad字段信息
                }
                coco_output['annotations'].append(annotation)
                annotation_id += 1
        
        image_id += 1

    # 保存到JSON文件
    with open(output_json, 'w') as json_file:
        json.dump(coco_output, json_file, indent=4)

    print(f'COCO格式文件已生成: {output_json}')

# 调整train和val的bbox并生成json文件
adjust_bbox_and_create_coco_json(img_ori_path_v_train, img_path_v_train, label_txt_path_train, json_train_path)
adjust_bbox_and_create_coco_json(img_ori_path_v_val, img_path_v_val, label_txt_path_val, json_val_path)