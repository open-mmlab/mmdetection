if '_base_':
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.data_preprocessors.data_preprocessor import BatchFixedSizePad
from mmcv.transforms.loading import LoadImageFromFile, LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomResize
from mmdet.datasets.transforms.transforms import RandomCrop, RandomFlip, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.sgd import SGD

norm_cfg = dict(type='BN', requires_grad=True)
image_size = (640, 640)
batch_augments = [dict(type=BatchFixedSizePad, size=image_size)]

model.merge(
    dict(
        data_preprocessor=dict(
            pad_size_divisor=64, batch_augments=batch_augments),
        backbone=dict(norm_cfg=norm_cfg, norm_eval=False),
        neck=dict(norm_cfg=norm_cfg),
        roi_head=dict(bbox_head=dict(norm_cfg=norm_cfg))))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=RandomResize,
        scale=image_size,
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(
        type=RandomCrop,
        crop_type='absolute_range',
        crop_size=image_size,
        allow_negative_crop=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=image_size, keep_ratio=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader.merge(
    dict(batch_size=8, num_workers=4, dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# learning policy
max_epochs = 50
train_cfg.merge(dict(max_epochs=max_epochs, val_interval=2))
param_scheduler = [
    dict(type=LinearLR, start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30, 40],
        gamma=0.1)
]

# optimizer
optim_wrapper.merge(
    dict(
        type=OptimWrapper,
        optimizer=dict(type=SGD, lr=0.08, momentum=0.9, weight_decay=0.0001),
        paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True),
        clip_grad=None))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=64))
