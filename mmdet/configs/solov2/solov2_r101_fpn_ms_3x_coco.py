if '_base_':
    from .solov2_r50_fpn_ms_3x_coco import *

# model settings
model.merge(
    dict(
        backbone=dict(
            depth=101, init_cfg=dict(checkpoint='torchvision://resnet101'))))
