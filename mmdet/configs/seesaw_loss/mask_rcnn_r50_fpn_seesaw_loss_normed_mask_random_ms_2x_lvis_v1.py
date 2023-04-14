if '_base_':
    from .mask_rcnn_r50_fpn_seesaw_loss_random_ms_2x_lvis_v1 import *
from mmdet.models.layers.normed_predictor import NormedConv2d

model.merge(
    dict(
        roi_head=dict(
            mask_head=dict(
                predictor_cfg=dict(type=NormedConv2d, tempearture=20)))))
