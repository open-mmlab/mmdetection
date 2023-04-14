if '_base_':
    from .detr_r50_8xb2_500e_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=18,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet18')),
        neck=dict(in_channels=[512])))
