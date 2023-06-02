_base_ = 'solov2_light_r50_fpn_3x_coco.py'

# model settings
model = dict(
    backbone=dict(
        depth=18, init_cfg=dict(checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
