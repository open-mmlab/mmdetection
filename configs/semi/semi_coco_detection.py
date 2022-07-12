# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
file_client_args = dict(backend='disk')

image_size = (200, 200)

color_space = [
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Invert')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='SolarizeAdd')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

geometric = [[dict(type='Rotate')], [dict(type='ShearX')],
             [dict(type='ShearY')], [dict(type='TranslateX')],
             [dict(type='TranslateY')]]

sup_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
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
                   'homography_matrix')),
]

strong_pipeline = [
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='RandAugment', aug_space=geometric, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
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
            ann_file='coco_semi_annos/instances_train2017.1@10.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=sup_pipeline,
        ),
        unsup=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='coco_semi_annos/'
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
