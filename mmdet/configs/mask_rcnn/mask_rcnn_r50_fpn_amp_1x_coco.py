if '_base_':
    from .mask_rcnn_r50_fpn_1x_coco import *
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper.merge(dict(type=AmpOptimWrapper))
