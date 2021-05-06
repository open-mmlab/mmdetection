checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 0
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/mobilenet_v2-2s_ssd-256x256'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True