_base_ = './rtdetr_r50vd_4xb4-72e_coco.py'
pretrained = 'https://github.com/flytocc/mmdetection/releases/download/model_zoo/resnet34vd_pretrained_f6a72dc5.pth'  # noqa

model = dict(
    backbone=dict(
        depth=34,
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 256, 512]),
    encoder=dict(expansion=0.5),
    decoder=dict(num_layers=4))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, to decay_multi=0.0
num_blocks_list = (3, 4, 6, 3)  # r34
downsample_norm_idx_list = (2, 3, 3, 3)  # r34
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
_base_.optim_wrapper.paramwise_cfg.custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.bn': backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list)
    for block_id in range(num_blocks)
})
_base_.optim_wrapper.paramwise_cfg.custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.downsample.{downsample_norm_idx - 1}': backbone_norm_multi   # noqa
    for stage_id, (num_blocks, downsample_norm_idx) in enumerate(zip(num_blocks_list, downsample_norm_idx_list))  # noqa
    for block_id in range(num_blocks)
})
