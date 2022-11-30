import json
import os
import shutil

from auto_training.utils.kitti_conversion import read_files, is_image


def parse_training_data_classes(train_annot_file_path):
    coco = json.loads(open(train_annot_file_path, "rb").read())
    cats = coco["categories"]
    return cats



def copy_images(src, target_folder):
    files = read_files(src)
    images = [f for f in files if is_image(f)]
    for image in images:
        os.makedirs(target_folder, exist_ok=True)
        image_name = os.path.basename(image)
        target_path = os.path.join(target_folder, image_name)
        shutil.copy(image, target_path)

copy_images("/data/tlt_training/tlt_bega/data/kitti_training", "data/tlt_training/tlt_bega/data/test123")

