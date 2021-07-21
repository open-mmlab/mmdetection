_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='YOLOX',
    backbone=dict(type='YOLOPAFPN', depth=0.33, width=0.5),
    neck=None,
    bbox_head=dict(
        type='YOLOXHead',
        width=0.5,
        num_classes=80
    ),
    # test
    test_cfg=dict(
        min_bbox_size=0,
        conf_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=1000)
)

img_norm_cfg = dict(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=(640, 640), pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    test=dict(pipeline=test_pipeline))
