if '_base_':
    from .cornernet_hourglass104_8xb6_210e_mstest_coco import *

train_dataloader.merge(dict(batch_size=5))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (10 GPUs) x (5 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=50))
