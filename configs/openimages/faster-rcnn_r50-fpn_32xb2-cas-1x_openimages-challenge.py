_base_ = ['faster_rcnn_r50_fpn_32x2_1x_openimages_challenge.py']

# Use ClassAwareSampler
train_dataloader = dict(
    sampler=dict(_delete_=True, type='ClassAwareSampler', num_sample_class=1))
