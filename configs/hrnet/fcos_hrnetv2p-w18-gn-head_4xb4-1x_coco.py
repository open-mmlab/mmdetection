_base_ = './fcos_hrnetv2p-w32-gn-head_4xb4-1x_coco.py'
model = dict(
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(18, 36)),
            stage3=dict(num_channels=(18, 36, 72)),
            stage4=dict(num_channels=(18, 36, 72, 144))),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w18')),
    neck=dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256))
