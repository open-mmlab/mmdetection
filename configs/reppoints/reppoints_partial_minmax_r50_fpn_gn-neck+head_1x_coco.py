_base_ = './reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py'
model = dict(bbox_head=dict(transform_method='partial_minmax'))
work_dir = './work_dirs/reppoints_partial_minmax_r50_fpn_1x'
