# dataset settings
dataset_type = 'Flickr30KDataset'
data_root = 'data/flickr30k/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='mdetr_annotations/final_flickr_separateGT_train.json',
        data_prefix=dict(img='flickr30k_images/')),
        pipeline=train_pipeline,
        backend_args=backend_args)
        #pipeline=train_pipeline,
        #backend_args=backend_args)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='mdetr_annotations/final_flickr_separateGT_val.json',
        data_prefix=dict(img='flickr30k_images/')))
        # pipeline=test_pipeline,
        # backend_args=backend_args)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='mdetr_annotations/final_flickr_separateGT_test.json',
        data_prefix=dict(img='flickr30k_images/')))
        # pipeline=test_pipeline,
        # backend_args=backend_args)

val_evaluator = dict(
    type='PhrGroMetric',
    flickr_path='data/flickr30k/annotations/',
    ann_file='data/flickr30k/mdetr_annotations/final_flickr_separateGT_val.json'
    )
test_evaluator = dict(
    type='PhrGroMetric',
    flickr_path='data/flickr30k/annotations/',
    ann_file='data/flickr30k/mdetr_annotations/final_flickr_separateGT_test.json'
    )
