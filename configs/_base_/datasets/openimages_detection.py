# dataset settings
dataset_type = 'OpenImagesDataset'
data_root = 'data/OpenImages/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 800), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    # TODO: find a better way to collect image_level_labels
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances', 'image_level_labels'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,  # workers_per_gpu > 0 may occur out of memory
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/oidv6-train-annotations-bbox.csv',
        data_prefix=dict(img='OpenImages/train/'),
        label_file='annotations/class-descriptions-boxable.csv',
        hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
        meta_file='annotations/train-image-metas.pkl',
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/validation-annotations-bbox.csv',
        data_prefix=dict(img='OpenImages/validation/'),
        label_file='annotations/class-descriptions-boxable.csv',
        hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
        meta_file='annotations/validation-image-metas.pkl',
        image_level_ann_file='annotations/validation-'
        'annotations-human-imagelabels-boxable.csv',
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='OpenImagesMetric',
    iou_thrs=0.5,
    ioa_thrs=0.5,
    use_group_of=True,
    get_supercategory=True)
test_evaluator = val_evaluator
