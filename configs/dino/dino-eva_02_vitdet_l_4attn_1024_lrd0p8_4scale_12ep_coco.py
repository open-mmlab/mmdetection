_base_ = './dino-4scale_r50_8xb2-12e_coco.py'

norm_cfg = dict(type='LN2d', requires_grad=True)

model = dict(
    backbone=dict(
        type='EVA02_ViT',
        img_size=1024,
        patch_size=16,
        window_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        use_act_checkpoint=False,
        drop_path_rate=0.4,
        window_block_indexes=(list(range(0, 5)) + list(range(6, 11)) +
                              list(range(12, 17)) + list(range(18, 23)))),
    neck=dict(
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=4,
        norm_cfg=norm_cfg,
    ))

custom_hooks = [dict(type='Fp16CompresssionHook')]
