if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from .._base_.datasets.wider_face import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from torch.optim.sgd import SGD
# model settings
model.merge(dict(bbox_head=dict(num_classes=1)))
# optimizer
optim_wrapper.merge(
    dict(optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001)))
