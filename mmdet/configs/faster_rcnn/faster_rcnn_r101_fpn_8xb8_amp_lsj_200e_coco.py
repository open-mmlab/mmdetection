if '_base_':
    from .faster_rcnn_r50_fpn_8xb8_amp_lsj_200e_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
