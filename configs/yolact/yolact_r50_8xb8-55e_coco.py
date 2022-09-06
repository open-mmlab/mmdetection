_base_ = 'yolact_r50_1xb8-55e_coco.py'

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(lr=8e-3),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning rate
max_epochs = 55
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20, 42, 49, 52],
        gamma=0.1)
]
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
