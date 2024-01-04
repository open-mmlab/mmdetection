_base_ = './rtdetr_r50vd_4xb4-72e_coco.py'

model = dict(
    backbone=dict(depth=101),
    neck=dict(out_channels=384),
    encoder=dict(
        in_channels=[384, 384, 384],
        out_channels=384,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=384),
            ffn_cfg=dict(embed_dims=384, feedforward_channels=2048))),
    decoder=dict(
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=384),
            cross_attn_cfg=dict(embed_dims=384),
            ffn_cfg=dict(embed_dims=384))),
    bbox_head=dict(embed_dims=384))

# set all layers in backbone to lr_mult=0.01
_base_.optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult = 0.01
