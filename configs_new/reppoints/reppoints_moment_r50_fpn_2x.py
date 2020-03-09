_base_ = [
    '../component/reppoints/reppoints_moment_r50_fpn_gn.py',
    '../component/coco_detection.py', '../component/schedule_2x.py',
    '../component/default_runtime.py'
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/reppoints_moment_r50_fpn_2x'
