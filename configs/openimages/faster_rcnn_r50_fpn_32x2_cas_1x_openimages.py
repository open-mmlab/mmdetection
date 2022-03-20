_base_ = ['faster_rcnn_r50_fpn_32x2_1x_openimages.py']

# Use ClassAwareSampler
data = dict(class_aware_sampler=dict(num_sample_class=1))
