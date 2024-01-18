_base_ = './rtdetr_r50vd_8xb2-72e_coco.py'
pretrained = 'https://github.com/flytocc/mmdetection/releases/download/model_zoo/resnet101vd_ssld_pretrained_64ed664a.pth'  # noqa

model = dict(
    backbone=dict(
        depth=101, init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(out_channels=384),
    encoder=dict(
        in_channels=[384, 384, 384],
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=384),
            ffn_cfg=dict(embed_dims=384, feedforward_channels=2048))))

# set all layers in backbone to lr_mult=0.01
_base_.optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult = 0.01
