_base_ = '../cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py'

model = dict(
    pretrained='open-mmlab://dla60',
    backbone=dict(
        type='DLANet',
        depth=60,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5))
