_base_ = 'faster-rcnn_r50-caffe_fpn_1x_coco.py'
max_iter = 90000

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=10000)
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=10000))
log_processor = dict(by_epoch=False)
