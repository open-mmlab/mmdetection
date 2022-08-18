_base_ = './cascade_rcnn_r50_fpn_lsj_200e_8x8_fp16_coco.py'

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
