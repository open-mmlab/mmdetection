_base_ = [
    'mmdet::_base_/datasets/coco_panoptic.py', 'mmdet::_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['projects.MaskDINO.maskdino'], allow_failed_imports=False)
image_size = (256, 256)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments)

num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='MaskDINO',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    panoptic_head=dict(
        type='MaskDINOHead',
        # from head
        num_classes=num_things_classes + num_stuff_classes,
        num_things_classes=num_things_classes,
        loss_weight=1,  # TODO: ???
        ignore_value=255,  # TODO ???
        encoder=dict(
            in_channels=[256, 512, 1024, 2048],
            in_strides=[4, 8, 16, 32],
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=2048,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm_cfg=dict(type='GN', num_groups=32),
            transformer_in_features=['res3', 'res4', 'res5'],
            common_stride=4,
            num_feature_levels=3,
            total_num_feature_levels=4,
            feature_order='low2high'),
        decoder=dict(
            in_channels=256,
            num_classes=num_things_classes+num_stuff_classes,
            hidden_dim=256,
            num_queries=300,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            mask_dim=256,
            enforce_input_project=False,
            two_stage=True,
            dn='seg',
            noise_scale=0.4,
            dn_num=20,
            initialize_box_type='no',
            initial_pred=True,
            learn_tgt=False,
            total_num_feature_levels=4,
            dropout=0.0,
            activation='relu',
            nhead=8,
            dec_n_points=4,
            mask_classification=True,
            return_intermediate_dec=True,
            query_dim=4,
            dec_layer_share=False,
            semantic_ce_loss=False),
        # from detector
        # sem_seg_head
        # criterion
        num_queries=300,
        object_mask_threshold=0.25,
        overlap_threshold=0.8,
        # metadata
        size_divisibility=32,
        sem_seg_postprocess_before_inference=True,
        # pixel_mean
        # pixel_std

        # # from Mask2Former
        # in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        # strides=[4, 8, 16, 32],
        # feat_channels=256,
        # out_channels=256,

    ),
    train_cfg=dict(  # corresponds to SetCriterion
        num_classes=num_things_classes + num_stuff_classes,
        matcher=dict(
            cost_class=4.0, cost_box=5.0, cost_giou=2.0,
            cost_mask=5.0, cost_dice=5.0, num_points=100),
        class_weight=4.0,
        box_weight=5.0,
        giou_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        dn='seg',
        dec_layers=9,
        box_loss=True,
        two_stage=True,
        eos_coef=0.1,
        num_points=100,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        semantic_ce_loss=False,
        panoptic_on=False,  # TODO: Why?
        deep_supervision=True
    ),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,  # TODO: Why False in Mask2Former-cocopan
        instance_on=True,
        # # max_per_image is for instance segmentation.
        # max_per_image=100,  # TODO: This arg is not in MaskDINO repo
        # iou_thr=0.8,  # TODO: This arg is not in MaskDINO repo
        # # In Mask2Former's panoptic postprocessing,
        # # it will filter mask area where score is less than 0.5 .
        # filter_low_score=True,  # TODO: This arg is not in MaskDINO repo
        test_topk_per_image=100,  # maybe max_per_image
        data_loader='coco_panoptic_lsj',  # d2 dataset mapper name
        pano_temp=0.06,
        focus_on_box=False,
        transform_eval=True,
        semantic_ce_loss=False
    ),
    init_cfg=None)

# dataset settings
data_root = 'data/coco/'
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

val_evaluator = [
    dict(
        type='CocoPanopticMetric',
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        seg_prefix=data_root + 'annotations/panoptic_val2017/',
    ),
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_val2017.json',
        metric=['bbox', 'segm'],
    )
]
test_evaluator = val_evaluator

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
max_iters = 368750
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[327778, 355092],
    gamma=0.1)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
interval = 5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=3,
        interval=interval))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
