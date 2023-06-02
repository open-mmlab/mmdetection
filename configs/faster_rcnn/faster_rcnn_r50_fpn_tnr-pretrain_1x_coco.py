_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

checkpoint = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)))

# `lr` and `weight_decay` have been searched to be optimal.
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.1,
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))
