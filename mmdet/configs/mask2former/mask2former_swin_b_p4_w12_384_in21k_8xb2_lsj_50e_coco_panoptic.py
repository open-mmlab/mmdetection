if '_base_':
    from .mask2former_swin_b_p4_w12_384_8xb2_lsj_50e_coco_panoptic import *

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'  # noqa

model.merge(
    dict(
        backbone=dict(
            init_cfg=dict(type='Pretrained', checkpoint=pretrained))))
