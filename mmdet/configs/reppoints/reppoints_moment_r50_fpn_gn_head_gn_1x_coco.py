if '_base_':
    from .reppoints_moment_r50_fpn_1x_coco import *

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model.merge(
    dict(neck=dict(norm_cfg=norm_cfg), bbox_head=dict(norm_cfg=norm_cfg)))
