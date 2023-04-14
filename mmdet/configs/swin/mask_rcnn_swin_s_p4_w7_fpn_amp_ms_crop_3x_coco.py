if '_base_':
    from .mask_rcnn_swin_t_p4_w7_fpn_amp_ms_crop_3x_coco import *

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model.merge(
    dict(
        backbone=dict(
            depths=[2, 2, 18, 2],
            init_cfg=dict(type='Pretrained', checkpoint=pretrained))))
