if '_base_':
    from .._base_.models.cascade_rcnn_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.backbones.detectors_resnet import DetectoRS_ResNet

model.merge(
    dict(
        backbone=dict(
            type=DetectoRS_ResNet,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True))))
