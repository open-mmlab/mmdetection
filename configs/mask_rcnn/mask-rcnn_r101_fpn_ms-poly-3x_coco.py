_base_ = [
    '../common/ms-poly_3x_coco-instance.py',
    '../_base_/models/mask-rcnn_r50_fpn.py'
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
