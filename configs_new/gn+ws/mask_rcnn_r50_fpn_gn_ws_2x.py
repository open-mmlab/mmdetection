_base_ = [
    '../component/mask_rcnn_r50_fpn.py', '../component/coco_instance.py',
    '../component/schedule_2x.py', '../component/default_runtime.py'
]
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='open-mmlab://jhu/resnet50_gn_ws',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    bbox_head=dict(
        _delete_=True,
        type='ConvFCBBoxHead',
        num_shared_convs=4,
        num_shared_fcs=1,
        in_channels=256,
        conv_out_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    mask_head=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg))
work_dir = './work_dirs/mask_rcnn_r50_fpn_gn_ws_2x'
