if '_base_':
    from .atss_r50_fpn_8xb8_amp_lsj_200e_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=18,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet18')),
        neck=dict(in_channels=[64, 128, 256, 512])))
