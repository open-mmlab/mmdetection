_base_ = './yolox_s.py'

# model settings
model = dict(
    backbone=dict(depth=0.67, width=0.75),
    bbox_head=dict(width=0.75)
)
