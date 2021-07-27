_base_ = './yolox_s.py'

model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, csp_num_blocks=2),
    bbox_head=dict(in_channels=192, feat_channels=192),
)
