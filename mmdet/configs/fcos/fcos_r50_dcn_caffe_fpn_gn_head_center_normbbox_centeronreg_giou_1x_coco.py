if '_base_':
    from .fcos_r50_caffe_fpn_gn_head_1x_coco import *
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.losses.iou_loss import GIoULoss
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

# model settings
model.merge(
    dict(
        data_preprocessor=dict(
            type=DetDataPreprocessor,
            mean=[103.530, 116.280, 123.675],
            std=[1.0, 1.0, 1.0],
            bgr_to_rgb=False,
            pad_size_divisor=32),
        backbone=dict(
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron2/resnet50_caffe')),
        bbox_head=dict(
            norm_on_bbox=True,
            centerness_on_reg=True,
            dcn_on_last_conv=True,
            center_sampling=True,
            conv_bias=True,
            loss_bbox=dict(type=GIoULoss, loss_weight=1.0)),
        # training and testing settings
        test_cfg=dict(nms=dict(type='nms', iou_threshold=0.6))))

# learning rate
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1.0 / 3.0,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper.merge(dict(clip_grad=None))
