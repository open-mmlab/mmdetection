if '_base_':
    from .mask_rcnn_r50_fpn_gn_ws_all_2x_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://jhu/resnet101_gn_ws'))))
