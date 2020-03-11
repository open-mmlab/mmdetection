_base_ = [
    '../component/faster_rcnn_r50_fpn_4conv1fc.py',
    '../component/coco_detection.py', '../component/schedule_1x.py',
    '../component/default_runtime.py'
]
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='open-mmlab://jhu/resnet50_gn_ws',
    backbone=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    bbox_head=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg))
work_dir = './work_dirs/faster_rcnn_r50_fpn_gn_ws_1x'
