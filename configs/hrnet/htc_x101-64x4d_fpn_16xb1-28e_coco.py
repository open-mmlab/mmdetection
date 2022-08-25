_base_ = '../htc/htc_x101_64x4d_fpn_16x1_20e_coco.py'

# learning policy
max_epochs = 28
train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[24, 27],
        gamma=0.1)
]
