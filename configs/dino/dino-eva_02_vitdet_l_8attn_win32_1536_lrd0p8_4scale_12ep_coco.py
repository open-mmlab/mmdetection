_base_ = './dino-4scale_r50_8xb2-12e_coco.py'

model = dict(
    type='SimpleFeaturePyramid',
    net=dict(
        type='EVA02_ViT',
        img_size=1536,
        patch_size=16,
        window_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        use_act_checkpoint=True,
        drop_path_rate=0.4,
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) +
                              list(range(6, 8)) + list(range(9, 11)) +
                              list(range(12, 14)) + list(range(15, 17)) +
                              list(range(18, 20)) + list(range(21, 23)))),
    square_pad=1536,
    init_cfg=dict(
        type='Pretrained', checkpoint='/path/to/eva02_L_pt_m38m_p14to16.pt'),
)
