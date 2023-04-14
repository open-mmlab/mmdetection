if '_base_':
    from .mask_rcnn_r50_fpn_sample1e_3_ms_2x_lvis_v0_5 import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
