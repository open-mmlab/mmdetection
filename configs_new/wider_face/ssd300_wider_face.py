_base_ = [
    '../component/ssd300.py', '../component/wider_face.py',
    '../component/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=2))
# optimizer
optimizer = dict(type='SGD', lr=0.012, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[16, 20])
# runtime settings
total_epochs = 24
log_config = dict(interval=1)
work_dir = './work_dirs/ssd300_wider'
