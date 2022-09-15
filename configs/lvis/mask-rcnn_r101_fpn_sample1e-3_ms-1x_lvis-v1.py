_base_ = './mask-rcnn_r50_fpn_sample1e-3_ms-1x_lvis-v1.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
