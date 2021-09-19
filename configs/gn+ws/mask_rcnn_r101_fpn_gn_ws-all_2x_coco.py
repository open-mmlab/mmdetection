_base_ = './mask_rcnn_r50_fpn_gn_ws-all_2x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://jhu/resnet101_gn_ws')))
