_base_ = './mask-rcnn_r50-fpn-gn-ws-all_2x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://jhu/resnet101_gn_ws')))
