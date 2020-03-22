_base_ = [
    '../component/mask_rcnn_r50_fpn.py', '../component/coco_instance.py',
    '../component/schedule_2x.py', '../component/default_runtime.py'
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
        norm_cfg=norm_cfg),
    mask_head=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg))
work_dir = './work_dirs/mask_rcnn_r50_fpn_gn_ws_2x'
