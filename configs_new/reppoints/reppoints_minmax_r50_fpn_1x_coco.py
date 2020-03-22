_base_ = './reppoints_moment_r50_fpn_1x.py'
model = dict(bbox_head=dict(transform_method='minmax'))
work_dir = './work_dirs/reppoints_minmax_r50_fpn_1x'
