if '_base_':
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.detectors.atss import ATSS
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.necks.fpn import FPN
from mmdet.models.necks.dyhead import DyHead
from mmdet.models.dense_heads.atss_head import ATSSHead
from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.iou_loss import GIoULoss
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.task_modules.assigners.atss_assigner import ATSSAssigner
from mmcv.transforms.loading import LoadImageFromFile, LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations, LoadAnnotations
from mmdet.datasets.transforms.transforms import Resize, RandomFlip, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs

model = dict(
    type=ATSS,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=128),
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=[
        dict(
            type=FPN,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type=DyHead,
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False)
    ],
    bbox_head=dict(
        type=ATSSHead,
        num_classes=80,
        in_channels=256,
        pred_kernel_size=1,  # follow DyHead official implementation
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type=AnchorGenerator,
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),  # follow DyHead official implementation
        bbox_coder=dict(
            type=DeltaXYWHBBoxCoder,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type=GIoULoss, loss_weight=2.0),
        loss_centerness=dict(
            type=CrossEntropyLoss, use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type=ATSSAssigner, topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# optimizer
optim_wrapper.merge(dict(optimizer=dict(lr=0.01)))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True, backend='pillow'),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True, backend='pillow'),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader
