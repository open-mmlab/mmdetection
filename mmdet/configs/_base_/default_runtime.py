# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.runner import LogProcessor
from mmengine.visualization import LocalVisBackend

from mmdet.engine.hooks import DetVisualizationHook
from mmdet.visualization import DetLocalVisualizer

default_scope = None

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, interval=1),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=DetVisualizationHook))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(
    type=DetLocalVisualizer, vis_backends=vis_backends, name='visualizer')
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
