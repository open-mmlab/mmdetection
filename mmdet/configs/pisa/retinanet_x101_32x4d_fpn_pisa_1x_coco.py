if '_base_':
    from ..retinanet.retinanet_x101_32x4d_fpn_1x_coco import *
from mmdet.models.dense_heads.pisa_retinanet_head import PISARetinaHead
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss

model.merge(
    dict(
        bbox_head=dict(
            type=PISARetinaHead,
            loss_bbox=dict(type=SmoothL1Loss, beta=0.11, loss_weight=1.0)),
        train_cfg=dict(isr=dict(k=2., bias=0.), carl=dict(k=1., bias=0.2))))
