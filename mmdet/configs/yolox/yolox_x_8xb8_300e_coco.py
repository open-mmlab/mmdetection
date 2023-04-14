if '_base_':
    from .yolox_s_8xb8_300e_coco import *

# model settings
model.merge(
    dict(
        backbone=dict(deepen_factor=1.33, widen_factor=1.25),
        neck=dict(
            in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
        bbox_head=dict(in_channels=320, feat_channels=320)))
