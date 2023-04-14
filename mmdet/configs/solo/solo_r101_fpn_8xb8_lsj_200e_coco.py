if '_base_':
    from .solo_r50_fpn_8xb8_lsj_200e_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
