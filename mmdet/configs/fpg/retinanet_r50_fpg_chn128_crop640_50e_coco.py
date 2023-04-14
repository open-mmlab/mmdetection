if '_base_':
    from .retinanet_r50_fpg_crop640_50e_coco import *

model.merge(
    dict(
        neck=dict(out_channels=128, inter_channels=128),
        bbox_head=dict(in_channels=128)))
