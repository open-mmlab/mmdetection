_base_ = ['../_base_/models/retinanet_r50_fpn.py', '../common/ms_3x_coco.py']
# optimizer
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01))
