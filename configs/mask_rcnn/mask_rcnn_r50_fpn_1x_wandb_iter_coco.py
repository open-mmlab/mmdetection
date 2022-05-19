_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[300, 600])

# Setup the runner
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=1000)
# Set evaluation interval
evaluation = dict(interval=20)
# Set checkpoint interval
checkpoint_config = dict(interval=20)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'mmdetection',
                'group': 'maskrcnn-r50-fpn-1x-coco'
             },
             interval=50,
             log_checkpoint=True,
             num_eval_images=100)
        ])
