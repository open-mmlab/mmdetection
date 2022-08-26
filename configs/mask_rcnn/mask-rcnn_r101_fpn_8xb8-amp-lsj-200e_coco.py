_base_ = './mask-rcnn_r50_fpn_8xb8-amp-lsj-200e_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
