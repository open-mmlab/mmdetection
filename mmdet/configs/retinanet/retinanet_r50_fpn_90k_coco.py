if '_base_':
    from .retinanet_r50_fpn_1x_coco import *
from mmengine.runner.loops import IterBasedTrainLoop
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.dataset.sampler import InfiniteSampler

# training schedule for 90k
train_cfg.merge(
    dict(
        _delete_=True,
        type=IterBasedTrainLoop,
        max_iters=90000,
        val_interval=10000))
# learning rate policy
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=90000,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]
train_dataloader.merge(dict(sampler=dict(type=InfiniteSampler)))
default_hooks.merge(dict(checkpoint=dict(by_epoch=False, interval=10000)))

log_processor.merge(dict(by_epoch=False))
