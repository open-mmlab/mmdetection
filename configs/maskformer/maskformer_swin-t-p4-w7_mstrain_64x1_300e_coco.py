_base_ = ['./maskformer_r50_mstrain_64x1_300e_coco.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='MaskFormer',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,  # ! 0.3
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(
        type='MaskFormerHead',
        in_channels=[96, 192, 384, 768],  # pass to pixel_decoder inside
        pixel_decoder=dict(
            type='PixelDecoder',
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU')),
        enforce_decoder_input_project=True))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=6.0e-05,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=1.0, decay_mult=1.0)}))

# learning policy
lr_config = dict(warmup_ratio=1.0e-06, warmup_iters=1500)
