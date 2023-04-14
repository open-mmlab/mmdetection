if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from ..common.ms_3x_coco import *
from torch.optim.sgd import SGD
# optimizer
optim_wrapper.merge(
    dict(optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001)))
