_base_ = './faster-rcnn_r50_fpn_1x_coco.py'

# MMEngine support the following two ways, users can choose
# according to convenience
# optim_wrapper = dict(type='AmpOptimWrapper')
_base_.optim_wrapper.type = 'AmpOptimWrapper'
