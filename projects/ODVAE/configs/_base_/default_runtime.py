# _base_/default_runtime.py - Base runtime configuration

default_scope = 'mmdet'

# Basic environment configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Visualization backends
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Logging configuration
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# Default settings
load_from = None
resume = False

# Common default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Basic training and testing loops
train_cfg = dict(type='EpochBasedTrainLoop', val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Backend arguments for data loading
backend_args = None
