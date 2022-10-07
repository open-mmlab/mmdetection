import dataclasses
import os.path

from custom.utils import read_files, replace_extention, KittiAnnotation, convert_kitti_files

KITTI_PATH_TRAIN = "/home/chrissikek/repos/data/train"
KITTI_PATH_VAL = "/home/chrissikek/repos/data/val"





cocos = convert_kitti_files(KITTI_PATH_TRAIN, KITTI_PATH_VAL)
import json

with open('/home/chrissikek/repos/coco_train.json', 'w') as fp:
    json.dump(cocos[0], fp)
with open('/home/chrissikek/repos/coco_val.json', 'w') as fp:
    json.dump(cocos[1], fp)
# print(kitti_annots)