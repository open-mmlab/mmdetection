_base_ = './scnet_r50_fpn_1x_coco.py'
# learning policy
max_epochs = 20
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 19],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs)
