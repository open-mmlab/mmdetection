_base_ = 'mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_lsj_100e_coco.py'
# use FP16
fp16 = dict(loss_scale=512.)
