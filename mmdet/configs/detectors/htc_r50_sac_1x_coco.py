if '_base_':
    from ..htc.htc_r50_fpn_1x_coco import *
from mmdet.models.backbones.detectors_resnet import DetectoRS_ResNet

model.merge(
    dict(
        backbone=dict(
            type=DetectoRS_ResNet,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True))))
