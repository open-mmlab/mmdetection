_base_ = ['./ld_gflv1_r101_r18_fpn_coco_1x.py']

model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    bbox_head=dict(
        type='LDHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        teacher_config='configs/gfl/gfl_r101_f\
        pn_dconv_c3-c5_mstrain_2x_coco.py',
        teacher_model='gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco.pth'))
