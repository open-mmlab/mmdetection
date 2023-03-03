_base_ = './faster-rcnn_r50_fpn_1x_coco.py'
_base_.optim_wrapper.type = 'AmpOptimWrapper'
