if '_base_':
    from .tood_r50_fpn_1x_coco import *

model.merge(dict(bbox_head=dict(anchor_type='anchor_based')))
