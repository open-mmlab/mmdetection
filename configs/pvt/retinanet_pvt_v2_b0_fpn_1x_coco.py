_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RetinaNet',
    pretrained='https://github.com/whai362/PVT/'
    'releases/download/v2/pvt_v2_b0.pth',
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        pretrain_img_size=224,
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=(8, 8, 4, 4),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=False),
    neck=dict(in_channels=[32, 64, 160, 256]))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
