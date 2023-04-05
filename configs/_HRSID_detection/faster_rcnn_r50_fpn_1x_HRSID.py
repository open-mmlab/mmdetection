_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_HRSID_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='initmodel/resnet50-19c8e357.pth')),
    roi_head=dict(
        bbox_head=dict(num_classes=1)),
)

data = dict(
    samples_per_gpu=8,  # 单个 GPU 的 Batch size
    workers_per_gpu=2)

evaluation = dict(
    save_best='auto',
    interval=1,
    metric='bbox')

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# workflow = [('train', 1), ('val', 1)]

checkpoint_config = dict(
    save_last=False, max_keep_ckpts=1)