_base_ = 'mmdet::_base_/default_runtime.py'
default_hooks = dict(
    # logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, max_keep_ckpts=1,
        interval=5000))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
find_unused_parameters = True
