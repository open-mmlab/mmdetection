_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]
visualizer = dict(vis_backends=vis_backends)

# MMEngine support the following two ways, users can choose
# according to convenience
# default_hooks = dict(checkpoint=dict(interval=4))
_base_.default_hooks.checkpoint.interval = 4

# train_cfg = dict(val_interval=2)
_base_.train_cfg.val_interval = 2
