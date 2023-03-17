_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/wider_face.py',
    '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=1))

max_epochs = 24
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 20],
        gamma=0.1)
]

optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
default_hooks = dict(logger=dict(interval=1))
log_processor = dict(window_size=1)
