_base_ = './detic_centernet2_r50_fpn_4x_lvis_boxsup.py'

model = dict(
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
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/'
            'download/v1.0.0/swin_base_patch4_window7_224_22k.pth')),
    neck=dict(in_channels=[256, 512, 1024]))

# backend = 'pillow'
backend_args = None

train_pipeline = [
    dict(
        type='RandomResize',
        scale=(896, 896),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(896, 896),
        recompute_bbox=True,
        allow_negative_crop=True)
]

# training schedule for 180k
max_iter = 180000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iter, val_interval=180000)

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))
