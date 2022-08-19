_base_ = 'mask-rcnn_r50-caffe-fpn-syncbn-all-rpn-2conv_lsj-100e_coco.py'

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')
