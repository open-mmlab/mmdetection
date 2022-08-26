_base_ = './sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
