if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.detectors.retinanet import RetinaNet
from mmdet.models.backbones.pvt import PyramidVisionTransformer
from torch.optim.adamw import AdamW

model.merge(
    dict(
        type=RetinaNet,
        backbone=dict(
            _delete_=True,
            type=PyramidVisionTransformer,
            num_layers=[2, 2, 2, 2],
            init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                          'releases/download/v2/pvt_tiny.pth')),
        neck=dict(in_channels=[64, 128, 320, 512])))
# optimizer
optim_wrapper.merge(
    dict(
        optimizer=dict(
            _delete_=True, type=AdamW, lr=0.0001, weight_decay=0.0001)))
