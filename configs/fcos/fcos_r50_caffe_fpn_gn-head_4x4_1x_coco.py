_base_ = './fcos_r50_caffe_fpn_4x4_1x_coco.py'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(bbox_head=dict(norm_cfg=norm_cfg))
