if '_base_':
    from .._base_.models.ssd300 import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_2x import *
    from .._base_.default_runtime import *
from mmcv.transforms.loading import LoadImageFromFile, LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations, LoadAnnotations
from mmdet.datasets.transforms.transforms import Expand, MinIoURandomCrop, Resize, RandomFlip, PhotoMetricDistortion, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs
from mmengine.dataset.dataset_wrapper import RepeatDataset
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.sgd import SGD
from mmdet.engine.hooks.num_class_check_hook import NumClassCheckHook
from mmdet.engine.hooks.checkloss_hook import CheckInvalidLossHook

# dataset settings
input_size = 300
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
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
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(input_size, input_size), keep_ratio=False),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader.merge(
    dict(
        batch_size=8,
        num_workers=2,
        batch_sampler=None,
        dataset=dict(
            _delete_=True,
            type=RepeatDataset,
            times=5,
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='annotations/instances_train2017.json',
                data_prefix=dict(img='train2017/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args))))
val_dataloader.merge(dict(batch_size=8, dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# optimizer
optim_wrapper.merge(
    dict(
        type=OptimWrapper,
        optimizer=dict(type=SGD, lr=2e-3, momentum=0.9, weight_decay=5e-4)))

custom_hooks = [
    dict(type=NumClassCheckHook),
    dict(type=CheckInvalidLossHook, interval=50, priority='VERY_LOW')
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=64))
