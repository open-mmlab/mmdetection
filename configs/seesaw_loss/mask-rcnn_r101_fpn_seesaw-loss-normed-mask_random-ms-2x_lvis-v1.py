_base_ = './mask-rcnn_r50_fpn_seesaw-loss-normed-mask_random-ms-2x_lvis-v1.py'  # noqa: E501
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
