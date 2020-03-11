_base_ = '../mask_rcnn_r50_fpn_1x.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True), norm_eval=False))
work_dir = './work_dirs/mask_rcnn_r50_fpn_sbn_1x'
