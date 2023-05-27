_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/objects365v1_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(backbone=dict(norm_cfg=norm_cfg), bbox_head=dict(num_classes=365))

# Using 8 GPUS while training
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

runner = dict(
    _delete_=True, type='IterBasedRunner', max_iters=1350000)  # 36 epochs
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=1.0 / 1000,
    step=[900000, 1200000])

checkpoint_config = dict(interval=150000)
evaluation = dict(interval=150000, metric='bbox')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
