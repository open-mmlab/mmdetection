evaluation = dict(interval=1, metric='mAP', save_best='mAP', rule='greater')
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
custom_hooks = [dict(type='CancelTrainingHook', interval=5), dict(type='EnsureCorrectBestCheckpointHook')]
runner = dict(meta=dict(exp_name='train'),
              max_epochs=30,
              type='EpochRunnerWithCancel')
