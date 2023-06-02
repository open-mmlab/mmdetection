_base_ = './yolof_r50-c5_8xb8-1x_coco.py'

# We implemented the iter-based config according to the source code.
# COCO dataset has 117266 images after filtering. We use 8 gpu and
# 8 batch size training, so 22500 is equivalent to
# 22500/(117266/(8x8))=12.3 epoch, 15000 is equivalent to 8.2 epoch,
# 20000 is equivalent to 10.9 epoch. Due to lr(0.12) is large,
# the iter-based and epoch-based setting have about 0.2 difference on
# the mAP evaluation value.

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=22500,
    val_interval=4500)

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=22500,
        by_epoch=False,
        milestones=[15000, 20000],
        gamma=0.1)
]
train_dataloader = dict(sampler=dict(type='InfiniteSampler'))
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2500))

log_processor = dict(by_epoch=False)
