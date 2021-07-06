_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../common/lsj_100e_coco_instance.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
head_norm_cfg = dict(type='NaiveSyncBN', stats_mode='N', requires_grad=True)
model = dict(
    backbone=dict(
        frozen_stages=-1, norm_eval=False, norm_cfg=norm_cfg, init_cfg=None),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=head_norm_cfg),
        mask_head=dict(norm_cfg=head_norm_cfg)))
# optimizer
# fp16 = dict(loss_scale=512., mode='dynamic')

custom_imports = dict(imports='mmdet.models.utils.naive_syncbn')
