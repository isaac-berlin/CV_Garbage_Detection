import json
import os
from PIL import Image

# Set paths for input and output directories
input_dir = 'C:\\Users\\isaac\\dev\\CV_Garbage_Detection\\Data\\test'
output_dir = 'C:\\Users\\isaac\\dev\\CV_Garbage_Detection\\New_Data\\test'

# Define the categories for the COCO dataset
cat_arr = ['Aluminium foil', 'Bottle cap', 'Bottle', 'Broken glass', 'Can', 'Carton', 
           'Cigarette', 'Cup', 'Lid', 'Other litter', 'Other plastic', 'Paper', 
           'Plastic bag - wrapper', 'Plastic container', 'Pop tab', 'Straw', 
           'Styrofoam piece', 'Unlabeled litter']
categories = [{"id": i, "name": cat_arr[i]} for i in range(len(cat_arr))]

# Initialize COCO dataset
coco_dataset = {
    "info": {
        "description": "Converted YOLO to COCO Dataset",
        "version": "1.0",
        "year": 2024,
        "contributor": "Isaac Berlin",
        "date_created": "2024-11-17"
    },
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": []
}

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for image_file in os.listdir(input_dir):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files
    
    image_path = os.path.join(input_dir, "images", image_file)
    annotation_path = os.path.join(input_dir, "labels", f'{image_file.split(".")[0]}.txt')
    
    if not os.path.exists(annotation_path):
        print(f"Annotation file missing for {image_file}. Skipping.")
        continue
    
    # Load the image and get dimensions
    image = Image.open(image_path)
    width, height = image.size
    
    # Add image metadata
    image_id = len(coco_dataset["images"])
    image_dict = {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": image_file
    }
    coco_dataset["images"].append(image_dict)
    
    # Load annotations
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    
    for ann in annotations:
        try:
            category_name, x, y, w, h = ann.strip().split()
            x, y, w, h = map(float, [x, y, w, h])
            category_id = next((cat['id'] for cat in categories if cat['name'] == category_name), None)
            if category_id is None:
                print(f"Unknown category '{category_name}' in {image_file}. Skipping.")
                continue
            
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            
            ann_dict = {
                "id": len(coco_dataset["annotations"]),
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(ann_dict)
        
        except ValueError:
            print(f"Invalid annotation format in {image_file}. Skipping this annotation.")
            continue

# Save COCO dataset
output_path = os.path.join(output_dir, 'annotations.json')
with open(output_path, 'w') as f:
    json.dump(coco_dataset, f)

print(f"COCO dataset saved to {output_path}")
