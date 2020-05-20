_base_ = './retinanet_r50_caffe_fpn_mstrain_1x_coco_loss-norm.py'
# learning policy
lr_config = dict(step=[28, 34])
total_epochs = 36
