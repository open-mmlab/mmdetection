_base_ = './dino-4scale_r50_8xb2-12e_coco.py'

model = dict(
        bbox_head=dict(
        loss_iou=dict(type='GIoULoss', loss_weight=2.0, normalize_bbox=True)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0, normalize_bbox=True)
            ])),
    )

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')