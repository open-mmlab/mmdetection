_base_ = './mask-rcnn_r50_fpn_1x_coco.py'

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')
