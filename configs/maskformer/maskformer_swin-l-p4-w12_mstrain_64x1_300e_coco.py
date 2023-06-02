_base_ = './maskformer_r50_mstrain_16x1_75e_coco.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=depths,
        num_heads=[6, 12, 24, 48],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(
        in_channels=[192, 384, 768, 1536],  # pass to pixel_decoder inside
        pixel_decoder=dict(
            _delete_=True,
            type='PixelDecoder',
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU')),
        enforce_decoder_input_project=True))

# weight_decay = 0.01
# norm_weight_decay = 0.0
# embed_weight_decay = 0.0
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
norm_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'norm': norm_multi,
    'absolute_pos_embed': embed_multi,
    'relative_position_bias_table': embed_multi,
    'query_embed': embed_multi
}

# optimizer
optimizer = dict(
    type='AdamW',
    lr=6e-5,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=True,
    step=[250],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1e-6,
    warmup_iters=1500)
runner = dict(type='EpochBasedRunner', max_epochs=300)
