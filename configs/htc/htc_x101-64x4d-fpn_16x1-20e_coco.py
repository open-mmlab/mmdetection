_base_ = './htc_x101-32x4d-fpn_16x1-20e_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        groups=64,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))
