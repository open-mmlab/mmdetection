_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../common/lsj_50e_coco_instance.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained=None,
    backbone=dict(
        frozen_stages=-1, zero_init_residual=False, norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))
# optimizer
# fp16 = dict(loss_scale=512., mode='dynamic')
