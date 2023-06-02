_base_ = '../mask_rcnn/mask-rcnn_x101-32x4d_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True), norm_eval=False))
