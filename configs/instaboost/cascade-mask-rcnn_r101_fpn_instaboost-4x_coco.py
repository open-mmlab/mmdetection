_base_ = './cascade-mask-rcnn_r50_fpn_instaboost-4x_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
