_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/default_runtime.py'
]

model_wrapper = dict(
    type='SemiBaseDetector',
    detector='${model}',
    semi_train_cfg=dict(
        sup_weight=1.0,
        unsup_weight=4.0,
        score_thr=0.9,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(infer_on='teacher'),
    data_preprocessor=dict(
        type='MultiDataPreprocessor',
        data_preprocessor='${model.data_preprocessor}',
    ))

dataset_type = 'CocoDataset'
data_root = '/home/SENSETIME/chenzeming.vendor/Datasets/data/coco/'
file_client_args = dict(backend='disk')

image_size = (200, 200)

sup_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='MultiBranch', sup=dict(type='PackDetInputs'))
]

weak_pipeline = [
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix', 'dataset')),
]

strong_pipeline = [
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_num=2),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix', 'dataset')),
]

unsup_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadPseudoAnnos'),
    dict(
        type='MultiBranch',
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler', batch_size=2, source_ratio=[1, 1]),
    dataset=dict(
        type='SemiDataset',
        sup=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='../coco_semi_annos/instances_train2017.1@10.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=sup_pipeline,
        ),
        unsup=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='../coco_semi_annos/'
            'instances_train2017.1@10-unlabeled.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=unsup_pipeline,
        )))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=180000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=10000))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook')]
load_from = '/home/SENSETIME/chenzeming.vendor/Documents/' \
            'gitlib/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
