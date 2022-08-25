_base_ = 'mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_lsj_100e_coco.py'

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')
