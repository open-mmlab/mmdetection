_base_ = './rtdetr_r50vd_4xb4-72e_coco.py'

model = dict(
    backbone=dict(
        depth=34,
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False),
    neck=dict(in_channels=[128, 256, 512]),
    encoder=dict(expansion=0.5),
    decoder=dict(num_layers=4))
