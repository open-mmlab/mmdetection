_base_ = './atss_r50-fpn_lsj-200e_8x8-fp16_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
