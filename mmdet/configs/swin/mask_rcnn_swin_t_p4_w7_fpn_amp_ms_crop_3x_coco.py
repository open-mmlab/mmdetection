if '_base_':
    from .mask_rcnn_swin_t_p4_w7_fpn_ms_crop_3x_coco import *
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper
# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper.merge(dict(type=AmpOptimWrapper))
