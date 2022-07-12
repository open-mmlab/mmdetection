_base_ = './htc_x101_32x4d_fpn_16x1_20e_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        groups=64,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))
