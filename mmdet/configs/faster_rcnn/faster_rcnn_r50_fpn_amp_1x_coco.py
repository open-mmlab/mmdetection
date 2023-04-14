if '_base_':
    from .faster_rcnn_r50_fpn_1x_coco import *
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper

# MMEngine support the following two ways, users can choose
# according to convenience
# optim_wrapper = dict(type=AmpOptimWrapper)
optim_wrapper.type = 'AmpOptimWrapper'
