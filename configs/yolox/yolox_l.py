_base_ = './yolox_s.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, csp_num_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))
