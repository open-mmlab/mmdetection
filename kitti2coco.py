__author__ = "TaekYoung Lee, Jimin Lee"
__version__ = "1.0.0"

import os
import json
import mmcv
from random import shuffle
from shutil import copy

label_dir = r"/home/jimin_2022si/mmdetection/kitti/training/label_2"
image_dir = r"/home/jimin_2022si/mmdetection/kitti/training/image_2"
names = r"/home/jimin_2022si/mmdetection/kitti/names.txt"

train_output_dir = r"/home/jimin_2022si/mmdetection/kitti/train"
val_output_dir = r"/home/jimin_2022si/mmdetection/kitti/val"
train_val_ratio = 0.8
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

def cvt_bbox(x1, y1, x2, y2):
    return [int(x1), int(y2), int(x2) - int(x1), int(y2) - int(y1)]

category_id = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
    "DontCare": 8,
}

categories = [
    {"id": 0, "name": "Car"},
    {"id": 1, "name": "Van"},
    {"id": 2, "name": "Truck"},
    {"id": 3, "name": "Pedestrian"},
    {"id": 4, "name": "Person_sitting"},
    {"id": 5, "name": "Cyclist"},
    {"id": 6, "name": "Tram"},
    {"id": 7, "name": "Misc"},
    {"id": 8, "name": "DontCare"},
]
 
train_output_dict = {
    "images": [],
    "annotations": [],
    "categories": categories,
}

val_output_dict = {
    "images": [],
    "annotations": [],
    "categories": categories,
}

annotation_id = 0
ids = []

with open(names) as f:
    ids = f.readlines()

shuffle(ids)

index = len(ids) * train_val_ratio

for id in ids[:index]:
    label_path = os.path.join(label_dir, id + ".txt")
    image_path = os.path.join(label_dir, id + ".png")
    height, width = mmcv.imread(image_path).shape[:2]
    
    copy(image_path, train_output_dir)

    image = {
        "id": int(id),
        "height": height,
        "width": width,
        "file_name": id,
    }
   
    with open(label_path) as file:
        for line in file.readlines():
            info = line.split(" ")
            
            annotation = {
                "id": annotation_id,
                "image_id": int(id),
                "category_id": category_id.get(info[0]),
                "area": 0.0,
                "bbox": cvt_bbox(info[4:8]),
                "iscrowd": 0,
                "segmentation": [],
            }
            
            annotation_id += 1
            train_output_dict["annotations"].append(annotation)
 
    train_output_dict["images"].append(image)

with open(os.path.join(train_output_dir, "train.json"), "w") as train:
    json.dump(train_output_dict, train)

for id in ids[index:]:
    label_path = os.path.join(label_dir, id + ".txt")
    image_path = os.path.join(label_dir, id + ".png")
    height, width = mmcv.imread(image_path).shape[:2]
 
    copy(image_path, val_output_dir)

    image = {
        "id": int(id),
        "height": height,
        "width": width,
        "file_name": id,
    }
   
    with open(label_path) as file:
        for line in file.readlines():
            info = line.split(" ")
            
            annotation = {
                "id": annotation_id,
                "image_id": int(id),
                "category_id": category_id.get(info[0]),
                "area": 0.0,
                "bbox": cvt_bbox(info[4:8]),
                "iscrowd": 0,
                "segmentation": [],
            }
            
            annotation_id += 1
            val_output_dict["annotations"].append(annotation)
 
    val_output_dict["images"].append(image)

with open(os.path.join(val_output_dir, "train.json"), "w") as val:
    json.dump(val_output_dict, val)

if __name__ == "__main__":
    pass
