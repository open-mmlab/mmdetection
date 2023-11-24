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
for k, v in _base_.optim_wrapper.paramwise_cfg.custom_keys.items():
    if 'backbone' in k:
        v['lr_mult'] = 0.01

# set all layers in backbone to lr_mult=0.1
# set all norm layers, to decay_multi=0.0
num_blocks_list = (3, 4, 23, 3)  # r101
downsample_norm_idx_list = (3, 3, 3, 3)  # r101
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = {'backbone': dict(lr_mult=0.1, decay_mult=1.0)}
custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.bn': backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.downsample.{downsample_norm_idx - 1}': backbone_norm_multi   # noqa
    for stage_id, (num_blocks, downsample_norm_idx) in enumerate(zip(num_blocks_list, downsample_norm_idx_list))  # noqa
    for block_id in range(num_blocks)
})
# optimizer
optim_wrapper = dict(paramwise_cfg=dict(custom_keys=custom_keys))
