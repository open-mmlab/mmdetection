if '_base_':
    from .yolact_r50_1xb8_55e_coco import *
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

# optimizer
optim_wrapper.merge(
    dict(
        type=OptimWrapper,
        optimizer=dict(lr=8e-3),
        clip_grad=dict(max_norm=35, norm_type=2)))
# learning rate
max_epochs = 55
param_scheduler = [
    dict(type=LinearLR, start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20, 42, 49, 52],
        gamma=0.1)
]
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=64))
