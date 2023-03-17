_base_ = 'mmdet::common/lsj-200e_coco-detection.py'

custom_imports = dict(
    imports=['projects.Detic.detic'], allow_failed_imports=False)

image_size = (1024, 1024)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]

cls_layer = dict(
    type='ZeroShotClassifier',
    zs_weight_path='rand',
    zs_weight_dim=512,
    use_bias=0.0,
    norm_weight=True,
    norm_temperature=50.0)
reg_layer = [
    dict(type='Linear', in_features=1024, out_features=1024),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=1024, out_features=4)
]

num_classes = 22047

model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=batch_augments),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    rpn_head=dict(
        type='CenterNetRPNHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        conv_bias=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    roi_head=dict(
        type='DeticRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=0,
                use_torchvision=True),
            out_channels=256,
            featmap_strides=[8, 16, 32],
            # approximately equal to
            # canonical_box_size=224, canonical_level=4 in D2
            finest_scale=112),
        bbox_head=[
            dict(
                type='DeticBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                cls_predictor_cfg=cls_layer,
                reg_predictor_cfg=reg_layer,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='DeticBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                cls_predictor_cfg=cls_layer,
                reg_predictor_cfg=reg_layer,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='DeticBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                cls_predictor_cfg=cls_layer,
                reg_predictor_cfg=reg_layer,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32],
            # approximately equal to
            # canonical_box_size=224, canonical_level=4 in D2
            finest_scale=112),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            class_agnostic=True,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.8,
                    neg_iou_thr=0.8,
                    min_pos_iou=0.8,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            score_thr=0.0001,
            nms_pre=1000,
            max_per_img=256,
            nms=dict(type='nms', iou_threshold=0.9),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5)))

backend = 'pillow'
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True, backend=backend),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01 * 4, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(norm_decay_mult=0.))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.00025,
        by_epoch=False,
        begin=0,
        end=4000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=25,
        by_epoch=True,
        milestones=[22, 24],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
