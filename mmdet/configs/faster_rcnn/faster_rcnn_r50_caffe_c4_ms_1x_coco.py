# Copyright (c) OpenMMLab. All rights reserved.
from mmcv import RandomChoiceResize
from mmcv.transforms import LoadImageFromFile
from mmengine.config import read_base

from mmdet.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                       RandomFlip)

with read_base():
    from .faster_rcnn_r50_caffe_c4_1x_coco import *

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=RandomChoiceResize,
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        resize_type=Resize,
        keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

train_dataloader.dataset.pipeline = train_pipeline
