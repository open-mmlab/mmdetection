_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

data_root = '/usr/videodate/dataset/subsetcoco/'
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
train_pipeline = [dict(type='DefaultFormatBundle'),
                  dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
                       meta_keys=('img_norm_cfg',))]

sub_dataset = dict(type="COCODataset",
                   ann_file=data_root + 'annotations/instances_train2017.json',
                   img_prefix=data_root + 'train2017/',
                   pipeline=None
                   )

train_dataset = dict(type="MosaicDetection",
                     dataset=sub_dataset,
                     ann_file=data_root + 'annotations/instances_train2017.json',
                     img_prefix=data_root + 'train2017/',
                     pipeline=train_pipeline)



data = dict(samples_per_gpu=4,
            workers_per_gpu=2,
            train=train_dataset,
            test=dict(type="COCODataset",
                      ann_file=data_root + 'annotations/instances_val2017.json',
                      img_prefix=data_root + 'val2017/',
                      pipeline=test_pipeline),
            val=dict(type="COCODataset",
                      ann_file=data_root + 'annotations/instances_val2017.json',
                      img_prefix=data_root + 'val2017/',
                      pipeline=test_pipeline),)
