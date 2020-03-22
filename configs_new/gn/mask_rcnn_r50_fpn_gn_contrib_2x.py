_base_ = [
    '../_base_/mask_rcnn_r50_fpn.py', '../_base_/coco_instance.py',
    '../_base_/schedule_2x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='open-mmlab://contrib/resnet50_gn',
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    bbox_head=dict(
        type='Shared4Conv1FCBBoxHead',
        conv_out_channels=256,
        norm_cfg=norm_cfg),
    mask_head=dict(norm_cfg=norm_cfg))
work_dir = './work_dirs/mask_rcnn_r50_fpn_gn_contrib_2x'
