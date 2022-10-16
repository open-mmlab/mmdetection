from custom.convert.utils import convert_kitti_files

KITTI_PATH_TRAIN = "/home/chrissikek/repos/data/brummer/train"
KITTI_PATH_VAL = "/home/chrissikek/repos/data/brummer/val"
OUT_PATH = "/home/chrissikek/repos/data/brummer"

# KITTI_PATH_TRAIN = "/home/chrissikek/repos/data/fst/221014_training_data/train"
# KITTI_PATH_VAL = "/home/chrissikek/repos/data/fst/221014_training_data/val"
# OUT_PATH = "/home/chrissikek/repos/data/fst/221014_training_data"




cocos = convert_kitti_files(KITTI_PATH_TRAIN, KITTI_PATH_VAL)
import json

with open(f'{OUT_PATH}/coco_train.json', 'w') as fp:
    json.dump(cocos[0], fp)
with open(f'{OUT_PATH}/coco_val.json', 'w') as fp:
    json.dump(cocos[1], fp)
# print(kitti_annots)