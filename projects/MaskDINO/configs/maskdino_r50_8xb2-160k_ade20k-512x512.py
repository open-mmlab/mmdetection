_base_ = [
    '../../../configs/_base_/datasets/ade20k_semantic.py',
    '../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.MaskDINO.maskdino'], allow_failed_imports=False)

image_size = (512, 512)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=False,
        pad_seg=True,
        seg_pad_value=255)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=False,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments)

num_classes = 150
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
        num_stuff_classes=num_classes,  # TODO
        num_things_classes=0,  # fake
        encoder=dict(
            in_channels=[256, 512, 1024, 2048],
            in_strides=[4, 8, 16, 32],
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,  # diff
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm_cfg=dict(type='GN', num_groups=32),
            transformer_in_features=['res3', 'res4', 'res5'],
            common_stride=4,
            num_feature_levels=3,
            total_num_feature_levels=3,  # diff
            feature_order='high2low2'),  # diff
        decoder=dict(
            in_channels=256,
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=100,  # diff
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            mask_dim=256,
            enforce_input_project=False,
            two_stage=False,  # diff
            dn='seg',
            noise_scale=0.4,
            dn_num=100,
            initialize_box_type='no',
            # initialize_box_type='mask2box',  # diff
            initial_pred=True,
            learn_tgt=False,
            total_num_feature_levels=3,  # diff
            dropout=0.0,
            activation='relu',
            nhead=8,
            dec_n_points=4,
            mask_classification=True,
            return_intermediate_dec=True,
            query_dim=4,
            dec_layer_share=False,
            semantic_ce_loss=True)),  # diff
    panoptic_fusion_head=dict(
        type='MaskDINOFusionHead',
        num_things_classes=0,  # fake
        num_stuff_classes=0,  # fake
        semantic_ce_loss=True,  # diff
        loss_panoptic=None,  # MaskDINOFusionHead has no training loss
        init_cfg=None),  # MaskDINOFusionHead has no module
    train_cfg=dict(  # corresponds to SetCriterion
        num_classes=num_classes,
        matcher=dict(
            cost_class=2.0, # diff
            cost_box=5.0,
            cost_giou=2.0,
            cost_mask=5.0,
            cost_dice=5.0,
            num_points=12544),
        class_weight=2.0,  # diff
        box_weight=5.0,
        giou_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        dn='seg',
        dec_layers=9,
        box_loss=True,
        two_stage=False,
        eos_coef=0.1,
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        semantic_ce_loss=True,  # diff
        panoptic_on=False,  # TODO: Why?
        deep_supervision=True),
    test_cfg=dict(
        panoptic_on=False,
        instance_on=False,
        semantic_on=True,
        panoptic_postprocess_cfg=dict(
            object_mask_thr=0.8,  # diff
            iou_thr=0.8,
            filter_low_score=
            True,  # it will filter mask area where score is less than 0.5.
            panoptic_temperature=0.06,
            transform_eval=True),
        instance_postprocess_cfg=dict(max_per_image=100, focus_on_box=False)),
    init_cfg=None)

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_mask=False,
        with_seg=True,
        reduce_zero_label=True),
    dict(
        type='RandomChoiceResize',
        scales=[int(image_size[0] * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=image_size, crop_type='absolute'),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'flip', 'flip_direction'))
]
train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True, backend='pillow'),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_mask=False,
        with_seg=True,
        reduce_zero_label=True),
    dict(
        type='PackDetInputs', meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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

max_iters = 160000
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[135000, 150000],
    gamma=0.1)

interval = 5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)

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
