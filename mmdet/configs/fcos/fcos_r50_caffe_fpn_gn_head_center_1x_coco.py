if '_base_':
    from .fcos_r50_caffe_fpn_gn_head_1x_coco import *

# model settings
model.merge(
    dict(bbox_head=dict(center_sampling=True, center_sample_radius=1.5)))
