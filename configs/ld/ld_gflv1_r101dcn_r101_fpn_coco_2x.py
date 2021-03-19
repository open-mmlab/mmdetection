_base_ = ['./ld_gflv1_r101_r18_fpn_coco_1x.py']

model = dict(
    pretrained='torchvision://resnet101',
    teacher_config='configs/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py',
    teacher_ckpt='gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco.pth',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5))
