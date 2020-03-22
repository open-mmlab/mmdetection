_base_ = './reppoints_moment_r50_fpn_1x.py'
lr_config = dict(step=[16, 22])
total_epochs = 24
work_dir = './work_dirs/reppoints_moment_r50_fpn_2x'
