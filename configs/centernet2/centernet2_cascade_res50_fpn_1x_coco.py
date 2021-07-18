_base_ = [
    '../_base_/models/centernet2_cascade_r50_fpn.py',
    '../_base_/datasets/coco_detection.py', #change your path to dataset
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
