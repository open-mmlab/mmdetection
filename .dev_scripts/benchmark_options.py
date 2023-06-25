# Copyright (c) OpenMMLab. All rights reserved.

third_part_libs = [
    'pip install -r ../requirements/albu.txt',
    'pip install instaboostfast',
    'pip install git+https://github.com/cocodataset/panopticapi.git',
    'pip install timm',
    'pip install mmpretrain',
    'pip install git+https://github.com/lvis-dataset/lvis-api.git',
    'pip install -r ../requirements/multimodal.txt',
    'pip install -r ../requirements/tracking.txt',
    'pip install git+https://github.com/JonathonLuiten/TrackEval.git',
]

default_floating_range = 0.5
model_floating_ranges = {'atss/atss_r50_fpn_1x_coco.py': 0.3}
