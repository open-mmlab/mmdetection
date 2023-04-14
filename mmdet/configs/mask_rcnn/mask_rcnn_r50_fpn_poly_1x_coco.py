if '_base_':
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.datasets.coco_instance import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import Resize, RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(
        type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs),
]
train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
