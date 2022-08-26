_base_ = '../mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py'
model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 4),
            stages=(False, True, True, True),
            position='after_conv3')
    ]))
