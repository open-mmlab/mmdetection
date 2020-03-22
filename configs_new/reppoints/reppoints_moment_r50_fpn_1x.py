_base_ = [
    '../_base_/reppoints_moment_r50_fpn_gn.py', '../_base_/coco_detection.py',
    '../_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
optimizer = dict(lr=0.01)
work_dir = './work_dirs/reppoints_moment_r50_fpn_1x'
