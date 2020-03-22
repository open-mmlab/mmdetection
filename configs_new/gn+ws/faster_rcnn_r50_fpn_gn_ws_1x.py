_base_ = [
    '../_base_/faster_rcnn_r50_fpn.py', '../_base_/coco_detection.py',
    '../_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='open-mmlab://jhu/resnet50_gn_ws',
    backbone=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    bbox_head=dict(
        type='Shared4Conv1FCBBoxHead',
        conv_out_channels=256,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg))
work_dir = './work_dirs/faster_rcnn_r50_fpn_gn_ws_1x'
