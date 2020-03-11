_base_ = [
    '../component/mask_rcnn_r50_fpn_4conv1fc.py',
    '../component/coco_instance.py', '../component/schedule_2x.py',
    '../component/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='open-mmlab://contrib/resnet50_gn',
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    bbox_head=dict(norm_cfg=norm_cfg),
    mask_head=dict(norm_cfg=norm_cfg))
work_dir = './work_dirs/mask_rcnn_r50_fpn_gn_contrib_2x'
