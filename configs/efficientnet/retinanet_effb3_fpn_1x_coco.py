_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=1 - 0.99)
model = dict(
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        stem_channels=40,
        norm_cfg=norm_cfg,
        out_indices=(2, 4, 6),
        scale=3,
        with_cp=True,
        dropout=0.2,
        init_cfg=dict(type='Pretrained', checkpoint='tensorflowtommcv.pth')),
    neck=dict(type='FPN', in_channels=[48, 136, 384], start_level=0),
)
