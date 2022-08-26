_base_ = './point-rend_r50-caffe_fpn_ms-1x_coco.py'

max_epochs = 36

# learning policy
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

train_cfg = dict(max_epochs=max_epochs)
