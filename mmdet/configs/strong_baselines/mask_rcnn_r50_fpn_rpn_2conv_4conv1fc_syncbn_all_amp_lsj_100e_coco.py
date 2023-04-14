if '_base_':
    from .mask_rcnn_r50_fpn_rpn_2conv_4conv1fc_syncbn_all_lsj_100e_coco import *
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper.merge(dict(type=AmpOptimWrapper))
