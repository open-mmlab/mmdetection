# data settings

dataset_type = 'CocoCaptionDataset'
data_root = 'data/coco/'

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

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='pillow',
        backend_args=backend_args),
    dict(
        type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id']),
]

# ann_file download from
# train dataset: https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json # noqa
# val dataset: https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json # noqa
# test dataset: https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json # noqa
# val evaluator: https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json # noqa
# test evaluator: https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json # noqa
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/coco_karpathy_val.json',
        pipeline=test_pipeline,
    ))

val_evaluator = dict(
    type='COCOCaptionMetric',
    ann_file=data_root + 'annotations/coco_karpathy_val_gt.json',
)

# # If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
