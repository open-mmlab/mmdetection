if '_base_':
    from ..mask_rcnn.mask_rcnn_r101_fpn_1x_coco import *

model.merge(
    dict(
        backbone=dict(plugins=[
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                stages=(False, True, True, True),
                position='after_conv3')
        ])))
