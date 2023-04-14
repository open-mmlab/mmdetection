if '_base_':
    from .faster_rcnn_r50_fpn_32xb2_1x_openimages import *
from mmdet.datasets.samplers.class_aware_sampler import ClassAwareSampler

# Use ClassAwareSampler
train_dataloader.merge(
    dict(
        sampler=dict(
            _delete_=True, type=ClassAwareSampler, num_sample_class=1)))
