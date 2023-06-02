_base_ = './reppoints-moment_r50_fpn-gn_head-gn_2x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
