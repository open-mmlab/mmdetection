_base_ = '../mask_rcnn_r50_fpn_1x.py'
model = dict(
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        gcb=dict(ratio=1. / 16., ),
        stage_with_gcb=(False, True, True, True)))
work_dir = './work_dirs/mask_rcnn_r16_gcb_c3-c5_r50_fpn_1x'
