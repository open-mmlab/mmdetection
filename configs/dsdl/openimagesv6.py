_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=601)))

# dsdl dataset settings

# please visit our platform [OpenDataLab](https://opendatalab.com/)
# to downloaded dsdl dataset.
dataset_type = 'DSDLDetDataset'
data_root = 'data/OpenImages'
train_ann = 'dsdl/set-train/train.yaml'
val_ann = 'dsdl/set-val/val.yaml'
specific_key_path = dict(
    image_level_labels='./image_labels/*/label',
    Label='./objects/*/label',
    is_group_of='./objects/*/isgroupof',
)

backend_args = dict(
    backend='petrel',
    path_mapping=dict({'data/': 's3://open_dataset_original/'}))

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
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances', 'image_level_labels'))
]

train_dataloader = dict(
    sampler=dict(type='ClassAwareSampler', num_sample_class=1),
    dataset=dict(
        type=dataset_type,
        with_imagelevel_label=True,
        with_hierarchy=True,
        specific_key_path=specific_key_path,
        data_root=data_root,
        ann_file=train_ann,
        filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        with_imagelevel_label=True,
        with_hierarchy=True,
        specific_key_path=specific_key_path,
        data_root=data_root,
        ann_file=val_ann,
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

default_hooks = dict(logger=dict(type='LoggerHook', interval=1000), )
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[1, 2],
        gamma=0.1)
]
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

val_evaluator = dict(
    type='OpenImagesMetric',
    iou_thrs=0.5,
    ioa_thrs=0.5,
    use_group_of=True,
    get_supercategory=True)

test_evaluator = val_evaluator
