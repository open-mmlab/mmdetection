_base_ = 'retinanet_r50_fpn_1x_coco.py'

# training schedule for 90k
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=90000,
    val_interval=10000)
# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=90000,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]
train_dataloader = dict(sampler=dict(type='InfiniteSampler'))
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=10000))

log_processor = dict(by_epoch=False)
