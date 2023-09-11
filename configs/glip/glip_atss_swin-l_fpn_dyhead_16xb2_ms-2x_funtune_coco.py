_base_ = './glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'

model = dict(
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        drop_path_rate=0.4,
    ),
    neck=dict(in_channels=[384, 768, 1536]),
    bbox_head=dict(early_fuse=True, num_dyhead_blocks=8, use_checkpoint=True))

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_l_mmdet-abfe026b.pth'  # noqa
