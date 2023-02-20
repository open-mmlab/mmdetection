# Copyright (c) OpenMMLab. All rights reserved.

third_part_libs = [
    'pip install -r ../requirements/albu.txt',
    'pip install instaboostfast',
    'pip install git+https://github.com/cocodataset/panopticapi.git',
    'pip install timm',
    'pip install mmcls>=1.0.0rc0',
    'pip install git+https://github.com/lvis-dataset/lvis-api.git',
]

default_floating_range = 0.5
model_floating_ranges = {'atss/atss_r50_fpn_1x_coco.py': 0.3}
