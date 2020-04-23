_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
# model settings
model = dict(
    type='FSAF',
    bbox_head=dict(
        type='FSAFHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            _delete_=True,
            type='TBLRBBoxCoder',
            normalizer=1.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='none'),
        loss_bbox=dict(
            _delete_=True,
            type='IoULoss',
            eps=1e-6,
            loss_weight=1.0,
            reduction='none'),
    ))

# training and testing settings
train_cfg = dict(
    assigner=dict(
        _delete_=True,
        type='EffectiveAreaAssigner',
        pos_area_thr=0.2,
        neg_area_thr=0.2,
        min_pos_iof=0.01),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=10, norm_type=2))
total_epochs = 13
