_base_ = 'mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco.py'
fp16 = dict(loss_scale=512.)
