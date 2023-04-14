if '_base_':
    from .queryinst_r50_fpn_300_proposals_crop_ms_480_800_3x_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
