_base_ = ['../gfl/gfl_r101_fpn_mstrain_2x_coco.py']

model = dict(
    bbox_head=dict(
        type='LDGFLHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        teacher_config=
        'configs/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py',
        teacher_model=
        'gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='LDLoss', loss_weight=0.25, T=10, alpha=1),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)))
custom_hooks = [dict(type='EpochHook')]

optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
)
find_unused_parameters = True