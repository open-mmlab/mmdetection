if '_base_':
    from .._base_.models.ssd300 import *
    from .._base_.datasets.wider_face import *
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_2x import *
from mmcv.transforms.loading import LoadImageFromFile, LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations, LoadAnnotations
from mmdet.datasets.transforms.transforms import PhotoMetricDistortion, Expand, MinIoURandomCrop, Resize, RandomFlip, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

model.merge(dict(bbox_head=dict(num_classes=1)))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PhotoMetricDistortion,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type=Expand,
        mean=model.data_preprocessor.mean,
        to_rgb=model.data_preprocessor.bgr_to_rgb,
        ratio_range=(1, 4)),
    dict(
        type=MinIoURandomCrop,
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type=Resize, scale=(300, 300), keep_ratio=False),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(300, 300), keep_ratio=False),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
train_dataloader.merge(
    dict(batch_size=32, num_workers=8, dataset=dict(pipeline=train_pipeline)))

val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type=MultiStepLR, by_epoch=True, milestones=[16, 20], gamma=0.1)
]

# optimizer
optim_wrapper.merge(
    dict(
        optimizer=dict(lr=0.012, momentum=0.9, weight_decay=5e-4),
        clip_grad=dict(max_norm=35, norm_type=2)))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (32 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=256))
