_base_ = './fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py'
model = dict(bbox_head=dict(center_sampling=True, center_sample_radius=1.5))
