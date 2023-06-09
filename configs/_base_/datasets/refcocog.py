# dataset settings
dataset_type = 'RefCOCODataset'
data_root = 'data/coco/'

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='LoadAnnotations',
        with_mask=True,
        with_bbox=False,
        with_seg=False,
        with_label=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'gt_masks', 'text'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train2014/'),
        ann_file='refcocog/instances.json',
        split_file='refcocog/refs(umd).p',
        split='val',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train2014/'),
        ann_file='refcocog/instances.json',
        split_file='refcocog/refs(umd).p',
        split='test',
        pipeline=test_pipeline))

# TODO: set the metrics
val_evaluator = dict(type='SimpleIoUMetric', label_key='gt_masks')
test_evaluator = val_evaluator
