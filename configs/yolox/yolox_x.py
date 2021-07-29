_base_ = './yolox_s.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, csp_num_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320))
