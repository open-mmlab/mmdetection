_base_ = 'mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco.py'

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')
