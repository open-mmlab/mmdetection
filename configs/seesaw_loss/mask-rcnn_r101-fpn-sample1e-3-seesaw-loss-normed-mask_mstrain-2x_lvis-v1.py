_base_ = './mask-rcnn_r50-fpn-sample1e-3-seesaw-loss-normed-mask_mstrain-2x_lvis-v1.py'  # noqa: E501
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
