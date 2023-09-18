_base_ = [
    '../../../configs/dino/dino-4scale_r50_8xb2-12e_coco.py',
]
custom_imports = dict(imports=['projects.DINO.dino02-dino'])
norm_cfg = dict(type='LN2d', requires_grad=True)

model = dict(
    type='DINO',
    num_queries=10,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        _delete_=True,
        type='DINOv2',
        img_size=518,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_size=37,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=dict(type='LN'),
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9,
                              10],  # global attention for 2, 5, 8, 11
        use_rel_pos=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./projects/DINO/configs/vit-base-p14_'\
                       'dinov2-pre_3rdparty_20230426-ba246503.pth'
             )),  # TODO: add checkpoint
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=4,
        norm_cfg=norm_cfg,
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
)

train_dataloader = dict(
    batch_size=1,
    # sampler=dict(type='DefaultSampler', shuffle=False),
)

custom_hooks = [dict(type='Fp16CompresssionHook')]
