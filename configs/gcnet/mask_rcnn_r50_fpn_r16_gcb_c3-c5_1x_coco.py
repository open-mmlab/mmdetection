_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        gcb=dict(ratio=1. / 16., ), stage_with_gcb=(False, True, True, True)))
