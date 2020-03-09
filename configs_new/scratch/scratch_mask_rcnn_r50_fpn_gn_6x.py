_base_ = [
    '../component/mask_rcnn_r50_fpn_gn.py', '../component/coco_instance.py',
    '../component/default_runtime.py'
]
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[65, 71])
total_epochs = 73
work_dir = './work_dirs/scratch_mask_rcnn_r50_fpn_gn_6x'
