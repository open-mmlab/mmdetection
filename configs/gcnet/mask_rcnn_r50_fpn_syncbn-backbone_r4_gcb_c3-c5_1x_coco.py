_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        gcb=dict(ratio=1. / 4., ),
        stage_with_gcb=(False, True, True, True)))
work_dir = './work_dirs/mask_rcnn_r4_gcb_c3-c5_r50_fpn_syncbn_1x'
