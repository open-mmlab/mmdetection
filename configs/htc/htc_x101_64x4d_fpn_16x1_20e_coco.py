_base_ = './htc_r50_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'))
data = dict(samples_per_gpu=1, workers_per_gpu=1)
# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
