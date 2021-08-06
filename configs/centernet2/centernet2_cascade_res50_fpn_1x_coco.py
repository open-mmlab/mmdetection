_base_ = [
    '../_base_/models/centernet2_cascade_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

# # lr_config = dict(
# #     policy='step',
# #     warmup='linear',
# #     warmup_iters=5000,
# #     warmup_ratio=0.00001,
# #     step=[18])
# # runner = dict(type='EpochBasedRunner', max_epochs=20)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    )

find_unused_parameters = True

workflow = [('train', 1), ('val', 1)]
