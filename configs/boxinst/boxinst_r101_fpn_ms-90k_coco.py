_base_ = './boxinst_r50_fpn_ms-90k_coco.py'

# model settings
model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
