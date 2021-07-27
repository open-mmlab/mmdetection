_base_ = './yolox_s.py'

# model settings
model = dict(
    backbone=dict(depth=1.33, width=1.25),
    bbox_head=dict(width=1.25)
)
