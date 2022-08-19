_base_ = './mask2former_r50_lsj-8x2-50e_coco-panoptic.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
