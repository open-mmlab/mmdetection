_base_ = './mask-rcnn_r50-fpn-gn-all_2x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet101_gn')))
