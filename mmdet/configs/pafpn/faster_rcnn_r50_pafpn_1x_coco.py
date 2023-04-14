if '_base_':
    from ..faster_rcnn.faster_rcnn_r50_fpn_1x_coco import *
from mmdet.models.necks.pafpn import PAFPN

model.merge(
    dict(
        neck=dict(
            type=PAFPN,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5)))
