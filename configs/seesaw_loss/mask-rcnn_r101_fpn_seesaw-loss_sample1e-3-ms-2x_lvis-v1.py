_base_ = './mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
