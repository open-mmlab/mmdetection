_base_ = './fcos_r50_caffe_fpn_gn_1x_4gpu_coco.py'
model = dict(bbox_head=dict(center_sampling=True, center_sample_radius=1.5))
work_dir = './work_dirs/fcos_center_r50_caffe_fpn_gn_1x_4gpu'
