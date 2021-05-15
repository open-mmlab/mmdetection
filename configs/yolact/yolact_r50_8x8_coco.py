_base_ = 'yolact_r50_1x8_coco.py'

optimizer = dict(type='SGD', lr=8e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[20, 42, 49, 52])
