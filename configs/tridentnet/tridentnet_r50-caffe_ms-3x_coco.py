_base_ = 'tridentnet_r50-caffe_ms-1x_coco.py'

# learning rate
max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[28, 34],
        gamma=0.1)
]
