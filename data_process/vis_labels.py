import json
import os

def filter_and_modify_coco_json(input_json_path, output_json_path, image_filenames):
    # Load the original COCO JSON file
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # Initialize the new COCO structure
    new_coco_data = {
        "images": [],
        "annotations": [],
        "categories": coco_data["categories"]  # Copy the categories directly
    }

    # Initialize new ID counters
    new_image_id = 1
    new_annotation_id = 1

    # Dictionary to map old image_id to new image_id
    image_id_mapping = {}

    # Create a set of image IDs that match the provided image filenames and assign new IDs
    for image in coco_data['images']:
        if image['file_name'] in image_filenames:
            image_id_mapping[image['id']] = new_image_id
            image['id'] = new_image_id
            new_coco_data['images'].append(image)
            new_image_id += 1

    # Filter annotations to include only those that match the image IDs and assign new IDs
    for annotation in coco_data['annotations']:
        if annotation['image_id'] in image_id_mapping:
            annotation['image_id'] = image_id_mapping[annotation['image_id']]
            annotation['id'] = new_annotation_id
            new_coco_data['annotations'].append(annotation)
            new_annotation_id += 1

    # Save the new COCO JSON to the output path
    with open(output_json_path, 'w') as f:
        json.dump(new_coco_data, f, indent=4)

    print(f'Filtered and modified annotations saved to {output_json_path}')

# # Example usage
# input_json_path = '/data3/pengpeiran/datasets/VTUAV/val_ir.json'  # Replace with your input JSON path
# output_json_path = '/data3/pengpeiran/datasets/VTUAV/vis.json'  # Replace with your desired output JSON path
# image_filenames = [
#     '00011.jpg',
#     '00631.jpg',
#     '00805.jpg',
#     '00924.jpg'
# ]  # Replace with your image filenames

# Example usage
input_json_path = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/val_rgb.json'  # Replace with your input JSON path
output_json_path = '/data3/pengpeiran/datasets/NII_CU_MAPD/4-channel/vis.json'  # Replace with your desired output JSON path
image_filenames = [
    'flight3_frame25031.jpg',
    'flight3_frame16361.jpg',
    'flight3_frame13061.jpg',
    'flight3_frame16951.jpg'
]  # Replace with your image filenames

filter_and_modify_coco_json(input_json_path, output_json_path, image_filenames)