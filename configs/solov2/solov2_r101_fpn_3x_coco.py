_base_ = 'solov2_r50_fpn_3x_coco.py'

# model settings
model = dict(
    backbone=dict(
        depth=101, init_cfg=dict(checkpoint='torchvision://resnet101')))
