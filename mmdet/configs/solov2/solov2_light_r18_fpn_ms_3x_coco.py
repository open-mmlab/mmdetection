if '_base_':
    from .solov2_light_r50_fpn_ms_3x_coco import *

# model settings
model.merge(
    dict(
        backbone=dict(
            depth=18, init_cfg=dict(checkpoint='torchvision://resnet18')),
        neck=dict(in_channels=[64, 128, 256, 512])))
