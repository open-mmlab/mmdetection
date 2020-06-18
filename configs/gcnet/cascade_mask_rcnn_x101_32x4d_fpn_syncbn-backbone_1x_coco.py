_base_ = '../cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True), norm_eval=False))
