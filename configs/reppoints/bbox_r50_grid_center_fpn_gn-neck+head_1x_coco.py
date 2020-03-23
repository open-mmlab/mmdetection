_base_ = './reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py'
model = dict(bbox_head=dict(transform_method='minmax', use_grid_points=True))
work_dir = './work_dirs/bbox_r50_grid_center_fpn_1x'
