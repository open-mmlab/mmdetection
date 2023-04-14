if '_base_':
    from .htc_x101_64x4d_fpn_16xb1_20e_coco import *
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs

model.merge(
    dict(
        backbone=dict(
            dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True))))

# dataset settings
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type=RandomResize, scale=[(1600, 400), (1600, 1400)], keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
