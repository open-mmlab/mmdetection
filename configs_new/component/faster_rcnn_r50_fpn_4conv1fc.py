_base_ = './faster_rcnn_r50_fpn.py'
model = dict(
    bbox_head=dict(type='Shared4Conv1FCBBoxHead', conv_out_channels=256))
