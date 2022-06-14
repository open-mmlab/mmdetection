_base_ = './detr_r50_8x2_150e_coco.py'

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    bbox_head=dict(in_channels=512))
