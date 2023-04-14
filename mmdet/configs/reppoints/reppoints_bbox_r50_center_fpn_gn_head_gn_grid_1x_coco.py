if '_base_':
    from .reppoints_moment_r50_fpn_gn_head_gn_1x_coco import *

model.merge(
    dict(bbox_head=dict(transform_method='minmax', use_grid_points=True)))
