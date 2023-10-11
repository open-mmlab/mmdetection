# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .mask2former_swin_b_p4_w12_384_8xb2_lsj_50e_coco_panoptic import *

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa

model.update(
    dict(
        backbone=dict(
            embed_dims=192,
            num_heads=[6, 12, 24, 48],
            init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
        panoptic_head=dict(num_queries=200, in_channels=[192, 384, 768,
                                                         1536])))

train_dataloader.update(dict(batch_size=1, num_workers=1))

# learning policy
max_iters = 737500
param_scheduler.update(dict(end=max_iters, milestones=[655556, 710184]))

# Before 735001th iteration, we do evaluation every 5000 iterations.
# After 735000th iteration, we do evaluation every 737500 iterations,
# which means that we do evaluation at the end of training.'
interval = 5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg.update(
    dict(
        max_iters=max_iters,
        val_interval=interval,
        dynamic_intervals=dynamic_intervals))
