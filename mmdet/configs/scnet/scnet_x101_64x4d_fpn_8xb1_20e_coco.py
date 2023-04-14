if '_base_':
    from .scnet_x101_64x4d_fpn_20e_coco import *

train_dataloader.merge(dict(batch_size=1, num_workers=1))

optim_wrapper.merge(dict(optimizer=dict(lr=0.01)))
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=8))
