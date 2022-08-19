_base_ = './mask-rcnn_r50-fpn-sample1e-3_mstrain-2x_lvis-v0.5.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
