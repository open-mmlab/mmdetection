_base_ = './lsj-100e_coco-instance.py'

# 8x25=200e
train_dataloader = dict(dataset=dict(times=8))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.067, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=25,
        by_epoch=True,
        milestones=[22, 24],
        gamma=0.1)
]
