_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model_wrapper = dict(
    type='SemiBaseDetector',
    detector='${model}',
    semi_train_cfg=dict(sup_weight=1.0, unsup_weight=4.0),
    semi_test_cfg=dict(infer_on='teacher'),
    data_preprocessor=dict(
        type='MultiDataPreprocessor',
        data_preprocessor='${model.data_preprocessor}',
    ))

file_client_args = dict(backend='disk')
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

sup_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='MultiBranch', sup=dict(type='PackDetInputs'))
]

weak_pipeline = [
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

strong_pipeline = [
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_num=2),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PackDetInputs'),
]

unsup_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiBranch',
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

train_dataloader = dict(
    batch_size=5,
    num_workers=5,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler', batch_size=5, source_ratio=[1, 4]),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='annotations/instances_train2017.json',
                data_prefix=dict(img='train2017/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=sup_pipeline,
            ),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='annotations/instances_train2017.json',
                data_prefix=dict(img='train2017/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=unsup_pipeline,
            )
        ],
    ))
