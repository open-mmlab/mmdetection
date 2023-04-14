if '_base_':
    from .faster_rcnn_r50_fpn_1x_coco import *
from mmdet.models.task_modules.samplers.ohem_sampler import OHEMSampler

model.merge(dict(train_cfg=dict(rcnn=dict(sampler=dict(type=OHEMSampler)))))
