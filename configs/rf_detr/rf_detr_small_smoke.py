_base_ = './rf_detr_small.py'

data_root = 'data/rf_detr_smoke/'
train_ann_file = 'annotations/train.json'
val_ann_file = 'annotations/val.json'
test_ann_file = val_ann_file
train_data_prefix = 'images/'
val_data_prefix = 'images/'
test_data_prefix = 'images/'

num_classes = 2
metainfo = dict(classes=('square', 'circle'))
image_size = 128

model = dict(
    num_classes=num_classes,
    model_config=dict(
        num_classes=num_classes,
        resolution=image_size,
        positional_encoding_size=image_size // 16,
        num_queries=10,
        num_select=10,
        group_detr=1,
        pretrain_weights=None),
    train_config=dict(
        dataset_dir=data_root,
        output_dir='work_dirs/rf_detr_small_smoke',
        group_detr=1,
        batch_size=1,
        grad_accum_steps=1,
        use_ema=False))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    batch_sampler=None,
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        metainfo=metainfo,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        metainfo=metainfo,
        pipeline=test_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = dict(ann_file=data_root + test_ann_file)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1, save_last=True))
