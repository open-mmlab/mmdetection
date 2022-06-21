_base_ = './cornernet_hourglass104_mstest_8x6_210e_coco.py'

train_dataloader = dict(batch_size=5)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (10 GPUs) x (5 samples per GPU)
auto_scale_lr = dict(base_batch_size=50)
