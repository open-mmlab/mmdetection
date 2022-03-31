_base_ = './scnet_x101_64x4d_fpn_20e_coco.py'
data = dict(samples_per_gpu=1, workers_per_gpu=1)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
