_base_ = './faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 23])
total_epochs = 24
