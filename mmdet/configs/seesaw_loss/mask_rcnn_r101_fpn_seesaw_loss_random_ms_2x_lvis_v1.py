if '_base_':
    from .mask_rcnn_r50_fpn_seesaw_loss_random_ms_2x_lvis_v1 import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
