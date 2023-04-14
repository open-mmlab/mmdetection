if '_base_':
    from .._base_.models.ssd300 import *
    from .._base_.datasets.voc0712 import *
    from .._base_.schedules.schedule_2x import *
    from .._base_.default_runtime import *
from mmcv.transforms.loading import LoadImageFromFile, LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations, LoadAnnotations
from mmdet.datasets.transforms.transforms import Expand, MinIoURandomCrop, Resize, RandomFlip, PhotoMetricDistortion, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs
from mmdet.engine.hooks.num_class_check_hook import NumClassCheckHook
from mmdet.engine.hooks.checkloss_hook import CheckInvalidLossHook
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.sgd import SGD
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

model.merge(
    dict(
        bbox_head=dict(
            num_classes=20,
            anchor_generator=dict(basesize_ratio_range=(0.2, 0.9)))))
# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
input_size = 300
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=Expand,
        mean=model.data_preprocessor.mean,
        to_rgb=model.data_preprocessor.bgr_to_rgb,
        ratio_range=(1, 4)),
    dict(
        type=MinIoURandomCrop,
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type=Resize, scale=(input_size, input_size), keep_ratio=False),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=PhotoMetricDistortion,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type=PackDetInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=(input_size, input_size), keep_ratio=False),
    # avoid bboxes being resized
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader.merge(
    dict(
        batch_size=8,
        num_workers=3,
        dataset=dict(  # RepeatDataset
            # the dataset is repeated 10 times, and the training schedule is 2x,
            # so the actual epoch = 12 * 10 = 120.
            times=10,
            dataset=dict(  # ConcatDataset
                # VOCDataset will add different `dataset_type` in dataset.metainfo,
                # which will get error if using ConcatDataset. Adding
                # `ignore_keys` can avoid this error.
                ignore_keys=['dataset_type'],
                datasets=[
                    dict(
                        type=dataset_type,
                        data_root=data_root,
                        ann_file='VOC2007/ImageSets/Main/trainval.txt',
                        data_prefix=dict(sub_data_root='VOC2007/'),
                        filter_cfg=dict(filter_empty_gt=True, min_size=32),
                        pipeline=train_pipeline),
                    dict(
                        type=dataset_type,
                        data_root=data_root,
                        ann_file='VOC2012/ImageSets/Main/trainval.txt',
                        data_prefix=dict(sub_data_root='VOC2012/'),
                        filter_cfg=dict(filter_empty_gt=True, min_size=32),
                        pipeline=train_pipeline)
                ]))))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

custom_hooks = [
    dict(type=NumClassCheckHook),
    dict(type=CheckInvalidLossHook, interval=50, priority='VERY_LOW')
]

# optimizer
optim_wrapper.merge(
    dict(
        type=OptimWrapper,
        optimizer=dict(type=SGD, lr=1e-3, momentum=0.9, weight_decay=5e-4)))

# learning policy
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 20],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=64))
