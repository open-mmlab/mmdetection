_base_ = ['faster_rcnn_r50_fpn_32x2_1x_openimages_challenge.py']

# Use ClassAwareSampler
data = dict(use_class_aware_sampler=True)
