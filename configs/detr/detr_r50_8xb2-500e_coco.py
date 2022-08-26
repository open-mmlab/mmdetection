_base_ = './detr_r50_8xb2-150e_coco.py'

# learning policy
max_epochs = 500
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[334],
        gamma=0.1)
]

# only keep latest 2 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
