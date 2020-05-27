_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1230), mask_head=dict(num_classes=1230)))
