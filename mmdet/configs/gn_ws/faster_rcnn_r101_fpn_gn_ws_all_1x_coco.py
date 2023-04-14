if '_base_':
    from .faster_rcnn_r50_fpn_gn_ws_all_1x_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://jhu/resnet101_gn_ws'))))
