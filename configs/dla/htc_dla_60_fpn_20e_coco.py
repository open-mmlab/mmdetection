_base_ = '../htc/htc_r50_fpn_1x_coco.py'

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

find_unused_parameters = True

# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
