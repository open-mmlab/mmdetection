_base_ = './retinanet_r50-caffe_fpn_ms-1x_coco.py'
# training schedule for 2x
train_cfg = dict(max_epochs=24)

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]
