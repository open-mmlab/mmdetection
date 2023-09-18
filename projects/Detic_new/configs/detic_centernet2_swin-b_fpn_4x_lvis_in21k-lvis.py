_base_ = './detic_centernet2_r50_fpn_4x_lvis_in21k-lvis.py'

image_size_det = (896, 896)
image_size_cls = (448, 448)

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
        convert_weights=True),
    neck=dict(in_channels=[256, 512, 1024]))

# training schedule for 180k
max_iter = 180000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iter, val_interval=180000)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))

load_from = './first_stage/detic_centernet2_swin-b_fpn_4x_lvis_boxsup.pth'

find_unused_parameters = True
