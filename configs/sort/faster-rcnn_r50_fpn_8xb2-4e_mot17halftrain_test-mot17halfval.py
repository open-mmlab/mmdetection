_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/mot_challenge_det.py', '../_base_/default_runtime.py'
]

model = dict(
    rpn_head=dict(
        bbox_coder=dict(clip_border=False),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
            bbox_coder=dict(clip_border=False),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0))),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'  # noqa: E501
    ))

# training schedule for 4e
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=4,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
