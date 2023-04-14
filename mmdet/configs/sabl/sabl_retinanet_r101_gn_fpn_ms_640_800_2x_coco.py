if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_2x import *
    from .._base_.default_runtime import *
from mmdet.models.dense_heads.sabl_retina_head import SABLRetinaHead
from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator, AnchorGenerator
from mmdet.models.task_modules.coders.bucketing_bbox_coder import BucketingBBoxCoder
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from mmdet.models.task_modules.assigners.approx_max_iou_assigner import ApproxMaxIoUAssigner
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs
from torch.optim.sgd import SGD
# model settings
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101')),
        bbox_head=dict(
            _delete_=True,
            type=SABLRetinaHead,
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            approx_anchor_generator=dict(
                type=AnchorGenerator,
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            square_anchor_generator=dict(
                type=AnchorGenerator,
                ratios=[1.0],
                scales=[4],
                strides=[8, 16, 32, 64, 128]),
            norm_cfg=norm_cfg,
            bbox_coder=dict(
                type=BucketingBBoxCoder, num_buckets=14, scale_factor=3.0),
            loss_cls=dict(
                type=FocalLoss,
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox_cls=dict(
                type=CrossEntropyLoss, use_sigmoid=True, loss_weight=1.5),
            loss_bbox_reg=dict(
                type=SmoothL1Loss, beta=1.0 / 9.0, loss_weight=1.5)),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type=ApproxMaxIoUAssigner,
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0.0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)))
# dataset settings
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=RandomResize, scale=[(1333, 480), (1333, 800)], keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
# optimizer
optim_wrapper.merge(
    dict(optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001)))
