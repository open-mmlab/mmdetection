_base_ = './mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')
