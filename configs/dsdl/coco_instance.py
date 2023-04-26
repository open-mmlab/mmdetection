_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../_base_/datasets/dsdl.py'
]

# dsdl dataset settings.

# please visit our platform [OpenDataLab](https://opendatalab.com/)
# to downloaded dsdl dataset.
data_root = 'data/COCO2017'
img_prefix = 'original'
train_ann = 'dsdl/set-train/train.yaml'
val_ann = 'dsdl/set-val/val.yaml'
specific_key_path = dict(ignore_flag='./annotations/*/iscrowd')

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]

train_dataloader = dict(
    dataset=dict(
        with_polygon=True,
        specific_key_path=specific_key_path,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img_path=img_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    dataset=dict(
        with_polygon=True,
        specific_key_path=specific_key_path,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img_path=img_prefix),
        pipeline=test_pipeline,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric', metric=['bbox', 'segm'], format_only=False)

test_evaluator = val_evaluator
