_base_ = [
    '../_base_/grid_rcnn/grid_rcnn_gn_head_r50_fpn.py',
    '../_base_/coco_detection.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3665,
    warmup_ratio=1.0 / 80,
    step=[17, 23])
total_epochs = 25
work_dir = './work_dirs/grid_rcnn_gn_head_r50_fpn_2x'
