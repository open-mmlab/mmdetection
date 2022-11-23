_base_ = ['../detr/detr_r50_8xb2-150e_coco.py']
group_detr = 11
model = dict(
    type='ConditionalDETR',
    num_query=300,
    group_detr=group_detr,
    decoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                cross_attn=False,
                group_detr=group_detr),
            cross_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.1, cross_attn=True))),
    bbox_head=dict(
        type='ConditionalDETRHead',
        loss_cls=dict(
            _delete_=True,
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GHungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ],
            group_detr=group_detr)))

# learning policy
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)

param_scheduler = [dict(type='MultiStepLR', end=50, milestones=[40])]
