if '_base_':
    from .solo_r50_fpn_3x_coco import *
from mmdet.models.dense_heads.solo_head import DecoupledSOLOHead
from mmdet.models.losses.dice_loss import DiceLoss
from mmdet.models.losses.focal_loss import FocalLoss

# model settings
model.merge(
    dict(
        mask_head=dict(
            type=DecoupledSOLOHead,
            num_classes=80,
            in_channels=256,
            stacked_convs=7,
            feat_channels=256,
            strides=[8, 8, 16, 32, 32],
            scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384,
                                                                      2048)),
            pos_scale=0.2,
            num_grids=[40, 36, 24, 16, 12],
            cls_down_index=0,
            loss_mask=dict(
                type=DiceLoss,
                use_sigmoid=True,
                activate=False,
                loss_weight=3.0),
            loss_cls=dict(
                type=FocalLoss,
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True))))
