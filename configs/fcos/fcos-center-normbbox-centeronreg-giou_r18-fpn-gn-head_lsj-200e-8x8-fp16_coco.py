_base_ = './fcos-center-normbbox-centeronreg-giou_r50-fpn-gn-head_lsj-200e-8x8-fp16_coco.py'  # noqa

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
