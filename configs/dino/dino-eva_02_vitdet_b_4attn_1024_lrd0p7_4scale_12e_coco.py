_base_ = './dino-4scale_r50_8xb2-12e_coco.py'

model=dict(
    type='SimpleFeaturePyramid',
    net=dict(
        type='EVA02_ViT',
        img_size=1024,
        patch_size=16,
        window_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4*2/3,
        use_act_checkpoint = False,
        drop_path_rate = 0.1,
        window_block_indexes = [0, 1, 3, 4, 6, 7, 9, 10]
    )
    square_pad=1024,
    init_cfg=dict(
            type='Pretrained',
            checkpoint='/path/to/eva02_B_pt_\
                in21k_p14to16.pt'
             ),
)