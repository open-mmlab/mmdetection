# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .dino_4scale_r50_8xb2_12e_coco import *

# from deformable detr hyper
model.update(
    dict(
        backbone=dict(frozen_stages=-1),
        bbox_head=dict(loss_cls=dict(loss_weight=2.0)),
        positional_encoding=dict(offset=-0.5, temperature=10000),
        dn_cfg=dict(group_cfg=dict(num_dn_queries=300))))

# optimizer
optim_wrapper.update(
    dict(
        optimizer=dict(lr=0.0002),
        paramwise_cfg=dict(
            custom_keys={
                'backbone': dict(lr_mult=0.1),
                'sampling_offsets': dict(lr_mult=0.1),
                'reference_points': dict(lr_mult=0.1)
            })))
