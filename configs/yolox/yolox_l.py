_base_ = './yolox_s.py'

# model settings
model = dict(
    backbone=dict(depth=1.0, width=1.0),
    bbox_head=dict(width=1.0)
)
