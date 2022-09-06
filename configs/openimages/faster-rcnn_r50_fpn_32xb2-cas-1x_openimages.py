_base_ = ['faster-rcnn_r50_fpn_32xb2-1x_openimages.py']

# Use ClassAwareSampler
train_dataloader = dict(
    sampler=dict(_delete_=True, type='ClassAwareSampler', num_sample_class=1))
