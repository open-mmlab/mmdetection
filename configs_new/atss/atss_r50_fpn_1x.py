_base_ = [
    '../_base_/atss/atss_r50_fpn.py', '../_base_/coco_detection.py',
    '../_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/atss_r50_fpn_1x'
