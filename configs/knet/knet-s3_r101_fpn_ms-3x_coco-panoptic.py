_base_ = './knet-s3_r50_fpn_ms-3x_coco-panoptic.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
