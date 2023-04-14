if '_base_':
    from .faster_rcnn_s50_fpn_syncbn_backbone_head_ms_range_1x_coco import *

model.merge(
    dict(
        backbone=dict(
            stem_channels=128,
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnest101'))))
