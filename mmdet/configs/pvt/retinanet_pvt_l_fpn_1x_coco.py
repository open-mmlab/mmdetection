if '_base_':
    from .retinanet_pvt_t_fpn_1x_coco import *
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper

model.merge(
    dict(
        backbone=dict(
            num_layers=[3, 8, 27, 3],
            init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                          'releases/download/v2/pvt_large.pth'))))
# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper.merge(dict(type=AmpOptimWrapper))
