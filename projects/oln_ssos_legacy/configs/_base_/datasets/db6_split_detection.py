dataset_type = 'VOSDB6SplitDataset'
data_root = '/media/capture/hdd8tb/dataset/dbf6/'

custom_imports = dict(
    imports=[
        'projects.oln_ssos_legacy.vos.datasets.vos_coco',
        'projects.oln_ssos_legacy.vos.datasets.pipelines.loading',
        'projects.oln_ssos_legacy.vos.datasets.pipelines.formatting'
    ],
    allow_failed_imports=False)

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotationsWithAnnID',
         with_bbox=True,
         with_label=True,
         with_ann_id=True,
         with_pseudo_labels=True,
         with_weak_bboxes=True,
         with_gt_bbox_ignore=True,
         ),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PseudoLabelFormatBundle',meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotationsWithAnnID', with_bbox=True),
    dict(
        type='PseudoLabelFormatBundle',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/train_db6_no_firearms.json',
        data_prefix=dict(img=data_root +  'images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        is_class_agnostic=True,
        train_class='id',
        eval_class='id',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/test_db6_no_firearms.json',
        data_prefix=dict(img=data_root + 'images/'),
        test_mode=True,
        pipeline=test_pipeline,
        is_class_agnostic=True,
        train_class='id',
        eval_class='id',
        backend_args=backend_args))
test_dataloader = val_dataloader


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test_db6_no_firearms.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

