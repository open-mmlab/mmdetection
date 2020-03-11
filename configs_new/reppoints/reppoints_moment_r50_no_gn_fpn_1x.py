_base_ = [
    '../component/reppoints/reppoints_moment_r50_fpn.py',
    '../component/coco_detection.py', '../component/schedule_1x.py',
    '../component/default_runtime.py'
]
optimizer = dict(lr=0.01)
work_dir = './work_dirs/reppoints_moment_r50_no_gn_fpn_1x'
