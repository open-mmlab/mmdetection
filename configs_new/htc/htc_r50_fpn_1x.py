_base_ = [
    '../_base_/models/htc_r50_fpn.py',
    '../_base_/datasets/coco_instance_semantic.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
work_dir = './work_dirs/htc_r50_fpn_1x'
