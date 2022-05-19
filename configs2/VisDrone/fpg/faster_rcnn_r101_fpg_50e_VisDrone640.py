_base_ = [
    '../../../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../dataset/VisDrone_detection_640x640_overlap40.py',
    '../schedule/schedule_50e.py',
    '../../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPG',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        inter_channels=256,
        num_outs=5,
        stack_times=9,
        paths=['bu'] * 9,
        same_down_trans=None,
        same_up_trans=dict(
            type='conv',
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        across_lateral_trans=dict(
            type='conv',
            kernel_size=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        across_down_trans=dict(
            type='interpolation_conv',
            mode='nearest',
            kernel_size=3,
            norm_cfg=norm_cfg,
            order=('act', 'conv', 'norm'),
            inplace=False),
        across_up_trans=None,
        across_skip_trans=dict(
            type='conv',
            kernel_size=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        output_trans=dict(
            type='last_conv',
            kernel_size=3,
            order=('act', 'conv', 'norm'),
            inplace=False),
        norm_cfg=norm_cfg,
        skip_inds=[(0, 1, 2, 3), (0, 1, 2), (0, 1), (0,), ()]),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)))
)
