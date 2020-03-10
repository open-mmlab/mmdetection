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
        norm_eval=False,
        norm_cfg=norm_cfg))
work_dir = './work_dirs/mask_rcnn_r50_fpn_sbn_1x'
