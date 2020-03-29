_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        plugin=dict(conv3=[dict(type='ContextBlock', ratio=1. / 4)]),
        stage_with_plugin=(False, True, True, True)))
