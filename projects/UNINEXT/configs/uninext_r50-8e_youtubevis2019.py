# dataset settings
_base_ = [
    'mmdet::_base_/datasets/youtube_vis.py', 'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.UNINEXT.uninext'], allow_failed_imports=False)

model = dict(
    type='UNINEXT_VID',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=4,
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,  # in uninext use freeze_at=0
        conv_cfg=dict(type='D2Conv2d'),
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapperBias',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        bias=True,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    tokenizer_cfg=dict(max_length=256, pad_max=True),
    text_encoder_cfg=dict(
        bert_name='bert-base-uncased',
        use_checkpoint=False,
        parallel_det=False,
        output_hidden_states=True,
        frozen_parameters=True),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=-0.5,  # -0.5 for uninext
        temperature=10000),  # 10000 for Uninext and original resnet dino
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    encoder=dict(
        vlfuse_num_layers=1,
        vlfuse_layer_cfg=dict(
            lang_model='bert-base-uncased',
            lang_dim=768,
            stable_softmax_2d=False,
            clamp_min_for_underflow=True,
            clamp_max_for_overflow=True,
            visiual_dim=256,
            embed_dim=2048,
            n_head=8,
            vlfuse_use_checkpoint=True),
        vision_num_layers=6,
        vision_layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    bbox_head=dict(
        type='UNINEXTHead',
        embed_dims=256,
        language_dims=768,
        repeat_nums=2,
        add_iou_branch=True,
        reid_head=dict(
            type='DeformableReidHead',
            num_layers=2,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
        matcher=dict(cost_bbox=5, cost_class=2, cost_giou=2, cost_mask=1),
    ),
    tracker=dict(
        type='IDOLTracker',
        init_score_thr=0.2,
        obj_score_thr=0.1,
        nms_thr_pre=0.5,
        nms_thr_post=0.05,
        addnew_score_thr=0.2,
        memo_tracklet_frames=10,
        memo_momentum=0.8,
        long_match=True,
        frame_weight=True,
        temporal_weight=True,
        memory_len=3,
        match_metric='bisoftmax'))

backend = 'pillow'
train_pipeline = None
test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', imdecode_backend=backend),
            dict(
                type='Resize',
                scale=(1333, 480),
                keep_ratio=True,
                backend=backend),
            dict(
                type='LangGuideDet',
                class_name=[
                    'person', 'giant_panda', 'lizard', 'parrot', 'skateboard',
                    'sedan', 'ape', 'dog', 'snake', 'monkey', 'hand', 'rabbit',
                    'duck', 'cat', 'cow', 'fish', 'train', 'horse', 'turtle',
                    'bear', 'motorbike', 'giraffe', 'leopard', 'fox', 'deer',
                    'owl', 'surfboard', 'airplane', 'truck', 'zebra', 'tiger',
                    'elephant', 'snowboard', 'boat', 'shark', 'mouse', 'frog',
                    'eagle', 'earless_seal', 'tennis_racket'
                ],
                train_state=False),
            # annotations is none
            dict(type='LoadTrackAnnotations', with_mask=True)
        ]),
    dict(type='PackVideoInputs', train_state=False)
]

train_dataloader = None
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='UNIYtvisDataSet',
        data_root=_base_.data_root,
        dataset_version=_base_.dataset_version,
        ann_file='annotations/instances_val_sub.json',
        data_prefix=dict(img_path='valid/JPEGImages'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='YouTubeVISUNIMetric',
    metric='youtube_vis_ap',
    outfile_prefix='./youtube_vis_results',
    format_only=True)
test_evaluator = val_evaluator

default_hooks = dict(
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')
