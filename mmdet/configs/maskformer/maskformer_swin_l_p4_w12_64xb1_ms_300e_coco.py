# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.optim.scheduler import LinearLR

from mmdet.models.backbones import SwinTransformer
from mmdet.models.layers import PixelDecoder

with read_base():
    from .maskformer_r50_ms_16xb1_75e_coco import *

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
depths = [2, 2, 18, 2]
model.update(
    dict(
        backbone=dict(
            _delete_=True,
            type=SwinTransformer,
            pretrain_img_size=384,
            embed_dims=192,
            patch_size=4,
            window_size=12,
            mlp_ratio=4,
            depths=depths,
            num_heads=[6, 12, 24, 48],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
            init_cfg=dict(type=PretrainedInit, checkpoint=pretrained)),
        panoptic_head=dict(
            in_channels=[192, 384, 768, 1536],  # pass to pixel_decoder inside
            pixel_decoder=dict(
                _delete_=True,
                type=PixelDecoder,
                norm_cfg=dict(type=GroupNorm, num_groups=32),
                act_cfg=dict(type=ReLU)),
            enforce_decoder_input_project=True)))

# optimizer

# weight_decay = 0.01
# norm_weight_decay = 0.0
# embed_weight_decay = 0.0
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
norm_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'norm': norm_multi,
    'absolute_pos_embed': embed_multi,
    'relative_position_bias_table': embed_multi,
    'query_embed': embed_multi
}

optim_wrapper.update(
    dict(
        optimizer=dict(lr=6e-5, weight_decay=0.01),
        paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0)))

max_epochs = 300

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[250],
        gamma=0.1)
]

train_cfg.update(dict(max_epochs=max_epochs))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (64 GPUs) x (1 samples per GPU)
auto_scale_lr.update(dict(base_batch_size=64))
