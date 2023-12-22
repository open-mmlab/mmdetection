_base_ = './dino-4scale_r50_8xb2-12e_coco.py'

norm_cfg = dict(type='LN2d', requires_grad=True)

model = dict(
    backbone=dict(
        type='EVA02_ViT',
        img_size=1024,
        patch_size=16,
        window_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4 * 2 / 3,
        use_act_checkpoint=False,
        drop_path_rate=0.1,
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10]),
    neck=dict(
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=4,
        norm_cfg=norm_cfg,
    ))

custom_hooks = [dict(type='Fp16CompresssionHook')]
