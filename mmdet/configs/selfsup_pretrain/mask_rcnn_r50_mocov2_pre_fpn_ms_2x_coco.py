if '_base_':
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.datasets.coco_instance import *
    from .._base_.schedules.schedule_2x import *
    from .._base_.default_runtime import *
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs

model.merge(
    dict(
        backbone=dict(
            frozen_stages=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='./mocov2_r50_800ep_pretrain.pth'))))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(type=RandomResize, scale=[(1333, 640), (1333, 800)], keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
