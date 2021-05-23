_base_ = [
    '../_base_/datasets/coco_instance.py', '../_base_/models/solo_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01)
