_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_HRSID_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]


model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='initmodel/resnet50-19c8e357.pth')),
    bbox_head=dict(num_classes=1)
)

data = dict(
    samples_per_gpu=8,  # 单个 GPU 的 Batch size
    workers_per_gpu=2)

evaluation = dict(
    save_best='auto',
    interval=1,
    metric='bbox')

checkpoint_config = dict(
    save_last=False, max_keep_ckpts=1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
