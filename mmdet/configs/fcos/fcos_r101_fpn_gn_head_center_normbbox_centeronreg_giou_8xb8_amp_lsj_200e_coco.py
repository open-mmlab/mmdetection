if '_base_':
    from .fcos_r50_fpn_gn_head_center_normbbox_centeronreg_giou_8xb8_amp_lsj_200e_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
