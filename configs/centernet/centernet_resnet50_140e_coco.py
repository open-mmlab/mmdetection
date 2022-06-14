_base_ = './centernet_resnet18_140e_coco.py'

model = dict(
    backbone=dict(
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(in_channel=2048))
