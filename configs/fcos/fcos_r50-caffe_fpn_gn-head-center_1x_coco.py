_base_ = './fcos_r50-caffe_fpn_gn-head_1x_coco.py'

# model settings
model = dict(bbox_head=dict(center_sampling=True, center_sample_radius=1.5))
