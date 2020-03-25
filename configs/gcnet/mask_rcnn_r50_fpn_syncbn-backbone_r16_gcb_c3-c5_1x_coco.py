_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        stage_with_gcb=(False, True, True, True),
        gcb=dict(ratio=1. / 16., )))
