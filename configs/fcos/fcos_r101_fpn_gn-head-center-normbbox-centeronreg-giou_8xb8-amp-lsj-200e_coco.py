_base_ = './fcos_r50_fpn_gn-head-center-normbbox-centeronreg-giou_8xb8-amp-lsj-200e_coco.py'  # noqa

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
