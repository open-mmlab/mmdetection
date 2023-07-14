num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='OneFormer',
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
        type='OneFormerHead',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        task='panoptic',
        max_seq_len=77,
        task_seq_len=77,
        task_mlp=dict(
            input_dim=77, hidden_dim=256, output_dim=256, num_layers=2),
        text_encoder=dict(
            context_length=77, width=256, layers=6, vocab_size=49408),
        text_projector=dict(
            input_dim=256, hidden_dim=256, output_dim=256, num_layers=2),
        prompt_ctx=dict(num_embeddings=16, embedding_dim=256),
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        class_transformer=dict(
            d_model=512,
            nhead=8,
            num_encoder_layers=0,
            num_decoder_layers=2,
            dim_feedforward=2048,
            dropout=0.1,
            normalize_before=False,
            return_intermediate_dec=False),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=None)
