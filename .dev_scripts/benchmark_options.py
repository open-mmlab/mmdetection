# Copyright (c) OpenMMLab. All rights reserved.

third_part_libs = [
    'pip install -r ../requirements/albu.txt', 'pip install instaboostfast',
    'pip install git+https://github.com/cocodataset/panopticapi.git'
]

default_floating_range = 0.5
model_floating_ranges = {'atss/atss_r50_fpn_1x_coco.py': 0.3}
