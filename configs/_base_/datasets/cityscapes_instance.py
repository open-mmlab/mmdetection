# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=[(2048, 800), (2048, 1024)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
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
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/instancesonly_filtered_gtFine_train.json',
            data_prefix=dict(img='leftImg8bit/train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instancesonly_filtered_gtFine_val.json',
        data_prefix=dict(img='leftImg8bit/val/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        metric=['bbox', 'segm']),
    dict(
        type='CityScapesMetric',
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        seg_prefix=data_root + '/gtFine/val',
        outfile_prefix='./work_dirs/cityscapes_metric/instance')
]

test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instancesonly_filtered_gtFine_test.json',
#         data_prefix=dict(img='leftImg8bit/test/'),
#         test_mode=True,
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=test_pipeline))
# test_evaluator = dict(
#         type='CityScapesMetric',
#         format_only=True,
#         outfile_prefix='./work_dirs/cityscapes_metric/test')
