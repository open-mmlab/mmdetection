_base_ = './point_rend_r50_caffe_fpn_mstrain_1x_coco.py'
# learning policy
lr_config = dict(step=[28, 34])
total_epochs = 36
