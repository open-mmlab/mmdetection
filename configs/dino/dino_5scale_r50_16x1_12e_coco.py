_base_ = './dino_4scale_r50_8x2_12e_coco.py'
model = dict(
    backbone=dict(out_indices=(0, 1, 2, 3)),
    neck=dict(in_channels=[256, 512, 1024, 2048], num_outs=5),
    bbox_head=dict(
        transformer=dict(
            num_feature_levels=5,
            encoder=dict(
                transformerlayers=dict(
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0))),
            decoder=dict(
                transformerlayers=dict(attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0),  # 0.1 for DeformDETR
                    dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0)  # 0.1 for DeformDETR
                ])))))
data = dict(samples_per_gpu=1)
