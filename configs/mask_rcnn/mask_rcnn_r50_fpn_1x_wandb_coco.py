_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# Set evaluation interval
evaluation = dict(interval=2)
# Set checkpoint interval
checkpoint_config = dict(interval=4)

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
             log_checkpoint_metadata=True,
             num_eval_images=100)
        ])
