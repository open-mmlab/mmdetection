_base_ = './mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
