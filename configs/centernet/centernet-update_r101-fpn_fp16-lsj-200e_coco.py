_base_ = './centernet-update_r50-fpn_fp16-lsj-200e_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
