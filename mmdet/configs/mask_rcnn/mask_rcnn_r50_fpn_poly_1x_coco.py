# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .._base_.datasets.coco_instance import *
    from .._base_.default_runtime import *
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(
        type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs),
]
train_dataloader.update(dict(dataset=dict(pipeline=train_pipeline)))
