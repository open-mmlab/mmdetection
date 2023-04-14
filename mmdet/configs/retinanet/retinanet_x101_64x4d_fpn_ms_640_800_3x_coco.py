if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from ..common.ms_3x_coco import *
from mmdet.models.backbones.resnext import ResNeXt
from torch.optim.sgd import SGD
# optimizer
model.merge(
    dict(
        backbone=dict(
            type=ResNeXt,
            depth=101,
            groups=64,
            base_width=4,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://resnext101_64x4d'))))
optim_wrapper.merge(dict(optimizer=dict(type=SGD, lr=0.01)))
