_base_ = './scnet_x101-64x4d_fpn_20e_coco.py'
train_dataloader = dict(batch_size=1, num_workers=1)

optim_wrapper = dict(optimizer=dict(lr=0.01))
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
