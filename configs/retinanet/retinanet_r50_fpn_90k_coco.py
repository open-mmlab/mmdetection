_base_ = 'retinanet_r50_fpn_1x_coco.py'

# training schedule for 90k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=90000, val_interval=10000)
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
default_hooks = dict(checkpoint=dict(interval=10000))
