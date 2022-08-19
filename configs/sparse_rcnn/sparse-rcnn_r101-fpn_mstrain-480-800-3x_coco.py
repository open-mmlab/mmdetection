_base_ = './sparse-rcnn_r50-fpn_mstrain-480-800-3x_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
