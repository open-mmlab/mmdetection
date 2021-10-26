_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=500)))

# dataset settings
dataset_type = 'OpenImagesChallengeDataset'
data_root = 'data/OpenImages/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, normed_bbox=True),
    dict(type='Resize', img_scale=(1024, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    # TODO: support
    # class_sample_path=data_root + "challenge2019/class_sample_train.pkl",
    # class_aware_sampling=True,
    train=dict(
        type=dataset_type,
        ann_file=data_root +
        'challenge2019/challenge-2019-train-detection-bbox.txt',
        img_prefix=data_root,
        label_csv_path=data_root + 'challenge2019/cls-label-description.csv',
        class_label_tree_path=data_root + 'challenge2019/class_label_tree.np',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'challenge2019/challenge-2019-validation-detection-bbox.txt',
        img_prefix=data_root,
        label_csv_path=data_root + 'challenge2019/cls-label-description.csv',
        class_label_tree_path=data_root + 'challenge2019/class_label_tree.np',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'challenge2019/challenge-2019-validation-detection-bbox.txt',
        img_prefix=data_root,
        label_csv_path=data_root + 'challenge2019/cls-label-description.csv',
        class_label_tree_path=data_root + 'challenge2019/class_label_tree.np',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
