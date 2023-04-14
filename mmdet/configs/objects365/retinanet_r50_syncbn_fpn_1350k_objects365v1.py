if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from .._base_.datasets.objects365v2_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmengine.runner.loops import IterBasedTrainLoop
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.sgd import SGD
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.dataset.sampler import InfiniteSampler

model.merge(
    dict(
        backbone=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
        bbox_head=dict(num_classes=365)))

# training schedule for 1350K
train_cfg.merge(
    dict(
        _delete_=True,
        type=IterBasedTrainLoop,
        max_iters=1350000,  # 36 epochs
        val_interval=150000))

# Using 8 GPUS while training
optim_wrapper.merge(
    dict(
        type=OptimWrapper,
        optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001),
        clip_grad=dict(max_norm=35, norm_type=2)))

# learning rate policy
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=10000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=1350000,
        by_epoch=False,
        milestones=[900000, 1200000],
        gamma=0.1)
]

train_dataloader.merge(dict(sampler=dict(type=InfiniteSampler)))
default_hooks.merge(dict(checkpoint=dict(by_epoch=False, interval=150000)))

log_processor.merge(dict(by_epoch=False))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=16))
