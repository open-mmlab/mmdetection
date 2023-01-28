_base_ = [
    '../_base_/models/ascend_retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# data
data = dict(samples_per_gpu=8)

# optimizer
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))

# Note: If the learning rate is set to 0.0025, the mAP will be 32.4.
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (1 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
