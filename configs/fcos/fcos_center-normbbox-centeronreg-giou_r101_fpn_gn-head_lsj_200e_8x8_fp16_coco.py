_base_ = './fcos_center-normbbox-centeronreg-giou_r50_fpn_gn-head_lsj_200e_8x8_fp16_coco.py'  # noqa

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
