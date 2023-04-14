if '_base_':
    from .yolov3_d53_8xb8_ms_608_273e_coco import *
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper
# fp16 settings
optim_wrapper.merge(dict(type=AmpOptimWrapper, loss_scale='dynamic'))
