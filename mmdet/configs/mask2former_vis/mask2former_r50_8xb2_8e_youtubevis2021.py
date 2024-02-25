# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.hooks import CheckpointHook
from mmengine.optim.scheduler.lr_scheduler import MultiStepLR
from mmengine.runner.loops import IterBasedTrainLoop

with read_base():
    from .mask2former_r50_8xb2_8e_youtubevis2019 import *

data_root = 'data/youtube_vis_2021/'
dataset_version = data_root[-5:-1]  # 2019 or 2021

train_dataloader.update(
    dataset=dict(
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2021_train.json'))

val_dataloader.update(
    dataset=dict(
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2021_valid.json'))
test_dataloader = val_dataloader

# learning policy
max_iters = 8000
param_scheduler = dict(
    type=MultiStepLR,
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[
        5500,
    ],
    gamma=0.1)
# runtime settings
train_cfg = dict(
    type=IterBasedTrainLoop, max_iters=max_iters, val_interval=8001)

default_hooks.update(
    checkpoint=dict(
        type=CheckpointHook, by_epoch=False, save_last=True, interval=500))
