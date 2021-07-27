_base_ = './yolox_tiny.py'

# model settings
model = dict(
    backbone=dict(depth=0.33, width=0.25, depthwise=True),
    bbox_head=dict(width=0.25, depthwise=True)
)



