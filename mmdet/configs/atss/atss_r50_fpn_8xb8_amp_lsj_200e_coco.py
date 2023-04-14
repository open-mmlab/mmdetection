if '_base_':
    from ..common.lsj_200e_coco_detection import *
from mmdet.models.data_preprocessors.data_preprocessor import BatchFixedSizePad, DetDataPreprocessor
from mmdet.models.detectors.atss import ATSS
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.necks.fpn import FPN
from mmdet.models.dense_heads.atss_head import ATSSHead
from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.iou_loss import GIoULoss
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.task_modules.assigners.atss_assigner import ATSSAssigner
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper
from torch.optim.sgd import SGD

image_size = (1024, 1024)
batch_augments = [dict(type=BatchFixedSizePad, size=image_size)]

model = dict(
    type=ATSS,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=batch_augments),
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type=FPN,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type=ATSSHead,
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type=AnchorGenerator,
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
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

train_dataloader.merge(dict(batch_size=8, num_workers=4))

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper.merge(
    dict(
        type=AmpOptimWrapper,
        optimizer=dict(
            type=SGD, lr=0.01 * 4, momentum=0.9, weight_decay=0.00004)))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=64))
