_base_ = '../mask_rcnn_r50_fpn_1x.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        gcb=dict(ratio=1. / 4., ),
        stage_with_gcb=(False, True, True, True),
        norm_eval=False,
        norm_cfg=norm_cfg))
work_dir = './work_dirs/mask_rcnn_r4_gcb_c3-c5_r50_fpn_syncbn_1x'
