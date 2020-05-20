_base_ = './retinanet_r50_caffe_fpn_mstrain_loss-ema-norm_1x_coco.py'
# learning policy
lr_config = dict(step=[28, 34])
total_epochs = 36
