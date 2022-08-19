_base_ = ['faster-rcnn_r50-fpn_32x2-1x_openimages-challenge.py']

# Use ClassAwareSampler
train_dataloader = dict(
    sampler=dict(_delete_=True, type='ClassAwareSampler', num_sample_class=1))
