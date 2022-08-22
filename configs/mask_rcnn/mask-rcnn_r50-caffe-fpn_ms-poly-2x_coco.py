_base_ = './mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

train_cfg = dict(max_epochs=24)
# learning rate
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
