import dataclasses
import os.path

from custom.utils import read_files, replace_extention, KittiAnnotation, convert_kitti_files

KITTI_PATH = "/home/chrissikek/repos/data/val"



files = [file for file in read_files(KITTI_PATH)]
path_dict = {os.path.basename(file):file for file in files}





coco = convert_kitti_files(files, path_dict)
import json
with open('/home/chrissikek/repos/coco_val.json', 'w') as fp:
    json.dump(coco, fp)
# print(kitti_annots)