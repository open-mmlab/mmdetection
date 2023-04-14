if '_base_':
    from .yolox_s_8xb8_300e_coco import *
from mmdet.models.data_preprocessors.data_preprocessor import BatchSyncRandomResize
from mmdet.datasets.transforms.transforms import Mosaic, RandomAffine, YOLOXHSVRandomAug, RandomFlip, Resize, Pad, Resize, Pad
from mmdet.datasets.transforms.loading import FilterAnnotations, LoadAnnotations
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs
from mmcv.transforms.loading import LoadImageFromFile

# model settings
model.merge(
    dict(
        data_preprocessor=dict(batch_augments=[
            dict(
                type=BatchSyncRandomResize,
                random_size_range=(320, 640),
                size_divisor=32,
                interval=10)
        ]),
        backbone=dict(deepen_factor=0.33, widen_factor=0.375),
        neck=dict(in_channels=[96, 192, 384], out_channels=96),
        bbox_head=dict(in_channels=96, feat_channels=96)))

img_scale = (640, 640)  # width, height

train_pipeline = [
    dict(type=Mosaic, img_scale=img_scale, pad_val=114.0),
    dict(
        type=RandomAffine,
        scaling_ratio_range=(0.5, 1.5),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    # Resize and Pad are for the last 15 epochs when Mosaic and
    # RandomAffine are closed by YOLOXModeSwitchHook.
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(
        type=Pad, pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type=FilterAnnotations, min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type=PackDetInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(416, 416), keep_ratio=True),
    dict(
        type=Pad, pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader
