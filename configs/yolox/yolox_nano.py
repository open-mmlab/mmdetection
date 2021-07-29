_base_ = './yolox_tiny.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=0.33, widen_factor=0.25, depthwise=True),
    neck=dict(
        in_channels=[64, 128, 256],
        out_channels=64,
        csp_num_blocks=1,
        depthwise=True),
    bbox_head=dict(in_channels=64, feat_channels=64, depthwise=True))
