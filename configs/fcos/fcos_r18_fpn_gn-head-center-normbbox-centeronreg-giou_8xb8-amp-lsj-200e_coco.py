_base_ = './fcos_center-normbbox-centeronreg-giou_r50_fpn_gn-head_lsj_200e_8x8_fp16_coco.py'  # noqa

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
