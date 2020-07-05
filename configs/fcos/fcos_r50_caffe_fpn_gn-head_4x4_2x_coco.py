_base_ = './fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py'

# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
