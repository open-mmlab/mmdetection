_base_ = './faster_rcnn_r50_fpn_lsj_200e_8x8_fp16_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
