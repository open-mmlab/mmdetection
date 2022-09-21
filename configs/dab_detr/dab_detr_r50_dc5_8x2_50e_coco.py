_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]
model = dict(
    type='DABDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        strides=(1, 2, 2, 1),
        dilations=(1, 1, 1, 2),  # dc5
        frozen_stages=1,
        norm_cfg=dict(
            type='FrozenBN2d',
            requires_grad=False),  # register torch.ops.FrozenBatchNorm2d
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='DABDETRHead',
        num_query=300,
        num_classes=80,  # default 80, 91 to align with official repo / debug
        in_channels=2048,
        iter_update=True,
        random_refpoints_xy=False,
        transformer=dict(
            type='DabDetrTransformer',
            num_patterns=0,
            encoder=dict(
                type='DabDetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    act_cfg=dict(type='PReLU'),  # default PReLU
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DabDetrTransformerDecoder',
                query_dim=4,
                num_layers=6,
                query_scale_type='cond_elewise',
                modulate_hw_attn=True,
                bbox_embed_diff_each_layer=False,
                transformerlayers=dict(
                    type='DabDetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='ConditionalAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    sa_dropout=0.1,
                    ca_dropout=0.1,
                    keep_query_pos=False,
                    act_cfg=dict(type='PReLU'),  # default PReLU
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        positional_encoding=dict(
            type='SinePositionalEncodingHW',
            num_feats=128,
            temperatureH=20,
            temperatureW=20,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),  # weights of cls cost and cls loss can't be different for DETRHead
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.,
                          eps=1e-8),  # default eps=1e-8
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,  # default batch size is 2 * 8, use 2 * 5
    workers_per_gpu=2,  # default 2, 0 for debug
    train=dict(
        continuous_categories=True,
        # default True, False to align with official repo
        pipeline=train_pipeline),
    val=dict(continuous_categories=True, pipeline=test_pipeline),
    test=dict(continuous_categories=True, pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
checkpoint_config = dict(interval=10)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step', step=[40])  # default 40 epochs, 1 epoch for debug train
runner = dict(
    type='EpochBasedRunner',
    max_epochs=50)  # default 50 epochs, 1 epoch for debug train
auto_scale_lr = dict(enable=True, base_batch_size=16)
