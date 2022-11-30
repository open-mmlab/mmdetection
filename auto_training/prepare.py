import argparse
import json
import os

from auto_training.utils.kitti_conversion import convert_kitti_files
from auto_training.utils.utils import copy_images



def prepare_folder(image_path, coco_path, coco, mode):
    train_folder = os.path.join(coco_path, mode)
    train_image_folder = os.path.join(train_folder, "image_2")
    os.makedirs(train_image_folder, exist_ok=True)
    copy_images(image_path, train_image_folder)
    with open(f'{train_folder}/coco_{mode}.json', 'w') as fp:
        json.dump(coco, fp)


def make_coco_folder(cocos, coco_path, train_image_path, val_image_path):
    prepare_folder(train_image_path, coco_path, cocos[0], mode="train")
    prepare_folder(val_image_path, coco_path, cocos[1], mode="val")



def main():
    args = parse_args()

    cocos = convert_kitti_files(args.kitti_train, args.kitti_val, args.target_class_map)
    make_coco_folder(cocos, args.coco_folder, args.kitti_train, args.kitti_val)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert kitti to coco dataset')
    parser.add_argument('kitti_train', type=str, help='train input data path, kitti dataset')
    parser.add_argument('kitti_val', type=str, help='val input data path, kitti dataset')
    parser.add_argument('coco_folder', type=str, help='output folder')
    parser.add_argument('--target-class-map', type=str, default="{}", help='target class mapping, json strin format. Map to None if class should not be used.')
    return parser.parse_args()


if __name__ == "__main__":
    main()