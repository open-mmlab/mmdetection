import os
import json
import random

from tqdm import tqdm

# Edit these paths
root_dir = "data/NEU/"
train_dir = os.path.join(root_dir, "train/")
val_dir = os.path.join(root_dir, "val/")
test_dir = os.path.join(root_dir, "test/")
save_dir = os.path.join(root_dir, "annotations/")

# Your classes
classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratch"]

# Example: random dummy box for each image
def create_dummy_annotation(img_id, w, h):
    return {
        "id": img_id,
        "image_id": img_id,
        "category_id": random.randint(1, 6),
        "bbox": [50, 50, w//2, h//2],
        "area": (w//2) * (h//2),
        "iscrowd": 0
    }

def images_from_folder(folder, start_id=1):
    files = os.listdir(folder)
    images = []
    annotations = []
    img_id = start_id
    for file in tqdm(files):
        if not file.endswith(".jpg"):
            continue
        images.append({
            "id": img_id,
            "width": 200,  # adjust
            "height": 200,  # adjust
            "file_name": file
        })
        # Example: dummy box — replace with your real box!
        annotations.append(create_dummy_annotation(img_id, 200, 200))
        img_id += 1
    return images, annotations

def make_json(images, annotations, output_file):
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i+1, "name": name} for i, name in enumerate(classes)]
    }
    with open(output_file, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"Saved: {output_file}")

# ---- Create folders ----
os.makedirs(save_dir, exist_ok=True)

# ---- Train ----
train_images, train_annots = images_from_folder(train_dir)
make_json(train_images, train_annots, os.path.join(save_dir, "train.json"))

# ---- Val ----
val_images, val_annots = images_from_folder(val_dir, start_id=len(train_images)+1)
make_json(val_images, val_annots, os.path.join(save_dir, "val.json"))

# ---- Test ----
test_images, test_annots = images_from_folder(test_dir, start_id=len(train_images)+len(val_images)+1)
make_json(test_images, test_annots, os.path.join(save_dir, "test.json"))