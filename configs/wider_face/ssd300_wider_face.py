_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/wider_face.py',
    '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=1))
# optimizer
optimizer = dict(type='SGD', lr=0.012, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 20])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
log_config = dict(interval=1)
