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
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
custom_hooks = [
    dict(type='CancelTrainingHook', interval=5),
    # dict(type='EnsureCorrectBestCheckpointHook')
]
runner = dict(meta=dict(exp_name='train'),
              max_epochs=0,  # Set via configurable parameters by the user.
              type='EpochRunnerWithCancel')
