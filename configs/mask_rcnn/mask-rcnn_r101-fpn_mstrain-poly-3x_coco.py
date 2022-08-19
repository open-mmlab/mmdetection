_base_ = [
    '../common/mstrain-poly_3x_coco_instance.py',
    '../_base_/models/mask-rcnn_r50-fpn.py'
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
