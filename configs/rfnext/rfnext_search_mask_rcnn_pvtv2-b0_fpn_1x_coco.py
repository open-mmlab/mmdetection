_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model setting
model = dict(
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(
            checkpoint=  # noqa
            'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth'  # noqa
        )),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 160, 256],
        out_channels=256,
        num_outs=5))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

custom_hooks = [
    dict(
        type='RFSearchHook',
        mode='search',
        rfstructure_file=None,
        verbose=True,
        by_epoch=True,
        config=dict(
            search=dict(
                step=0,
                max_step=11,
                search_interval=1,
                exp_rate=0.5,
                init_alphas=0.01,
                mmin=1,
                mmax=24,
                num_branches=2,
                skip_layer=[])))
]
