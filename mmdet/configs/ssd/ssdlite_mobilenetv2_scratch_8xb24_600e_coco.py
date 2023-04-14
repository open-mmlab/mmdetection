if '_base_':
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.backbones.mobilenet_v2 import MobileNetV2
from mmdet.models.necks.ssd_neck import SSDNeck
from torch.nn.modules.activation import ReLU6, ReLU6
from mmdet.models.dense_heads.ssd_head import SSDHead
from mmdet.models.task_modules.prior_generators.anchor_generator import SSDAnchorGenerator
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner
from mmdet.models.task_modules.samplers.pseudo_sampler import PseudoSampler
from mmcv.transforms.loading import LoadImageFromFile, LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations, LoadAnnotations
from mmdet.datasets.transforms.transforms import Expand, MinIoURandomCrop, Resize, RandomFlip, PhotoMetricDistortion, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs
from mmengine.dataset.dataset_wrapper import RepeatDataset
from mmengine.optim.scheduler.lr_scheduler import LinearLR, CosineAnnealingLR
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.sgd import SGD
from mmdet.engine.hooks.num_class_check_hook import NumClassCheckHook
from mmdet.engine.hooks.checkloss_hook import CheckInvalidLossHook

# model settings
data_preprocessor = dict(
    type=DetDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1)
model = dict(
    type=SingleStageDetector,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=MobileNetV2,
        out_indices=(4, 7),
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03)),
    neck=dict(
        type=SSDNeck,
        in_channels=(96, 1280),
        out_channels=(96, 1280, 512, 256, 256, 128),
        level_strides=(2, 2, 2, 2),
        level_paddings=(1, 1, 1, 1),
        l2_norm_scale=None,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type=ReLU6),
        init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03)),
    bbox_head=dict(
        type=SSDHead,
        in_channels=(96, 1280, 512, 256, 256, 128),
        num_classes=80,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type=ReLU6),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001),

        # set anchor size manually instead of using the predefined
        # SSD300 setting.
        anchor_generator=dict(
            type=SSDAnchorGenerator,
            scale_major=False,
            strides=[16, 32, 64, 107, 160, 320],
            ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
            min_sizes=[48, 100, 150, 202, 253, 304],
            max_sizes=[100, 150, 202, 253, 304, 320]),
        bbox_coder=dict(
            type=DeltaXYWHBBoxCoder,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=MaxIoUAssigner,
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        sampler=dict(type=PseudoSampler),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
env_cfg.merge(dict(cudnn_benchmark=True))

# dataset settings
input_size = 320
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=Expand,
        mean=data_preprocessor['mean'],
        to_rgb=data_preprocessor['bgr_to_rgb'],
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
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader.merge(
    dict(
        batch_size=24,
        num_workers=4,
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
                pipeline=train_pipeline))))
val_dataloader.merge(dict(batch_size=8, dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# training schedule
max_epochs = 120
train_cfg.merge(dict(max_epochs=max_epochs, val_interval=5))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=CosineAnnealingLR,
        begin=0,
        T_max=max_epochs,
        end=max_epochs,
        by_epoch=True,
        eta_min=0)
]

# optimizer
optim_wrapper.merge(
    dict(
        type=OptimWrapper,
        optimizer=dict(type=SGD, lr=0.015, momentum=0.9, weight_decay=4.0e-5)))

custom_hooks = [
    dict(type=NumClassCheckHook),
    dict(type=CheckInvalidLossHook, interval=50, priority='VERY_LOW')
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (24 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=192))
