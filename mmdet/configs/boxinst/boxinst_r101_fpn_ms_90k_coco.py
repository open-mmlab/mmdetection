if '_base_':
    from .boxinst_r50_fpn_ms_90k_coco import *

# model settings
model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
