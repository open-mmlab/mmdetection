_base_ = ['./oneformer_swin-t-p4-w7-224_panoptic.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=depths,
        num_heads=[6, 12, 24, 48],
        window_size=12,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(
        num_queries=150, in_channels=[192, 384, 768, 1536], task='panoptic'),
)
