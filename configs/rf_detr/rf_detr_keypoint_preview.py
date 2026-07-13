# Copyright (c) OpenMMLab. All rights reserved.
_base_ = './rf_detr_small.py'

data_root = 'data/coco/'
train_ann_file = 'annotations/person_keypoints_train2017.json'
val_ann_file = 'annotations/person_keypoints_val2017.json'
test_ann_file = val_ann_file

num_classes = 90
num_keypoints_per_class = [17]
image_size = 576

model = dict(
    model_config_type='keypoint_preview',
    train_config_type='keypoint',
    num_classes=num_classes,
    model_config=dict(
        num_classes=num_classes,
        resolution=image_size,
        positional_encoding_size=image_size // 12,
        use_grouppose_keypoints=True,
        dual_projector=True,
        dual_projector_kp_only=True,
        num_keypoints_per_class=num_keypoints_per_class,
        keypoint_cross_attn=True,
        inter_instance_kp_attn=False,
        grouppose_keypoint_dim_downscale=1,
        num_queries=100,
        num_select=100,
        pretrain_weights=None),
    train_config=dict(output_dir='work_dirs/rf_detr_keypoint_preview'))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=0)))
val_dataloader = dict(dataset=dict(ann_file=val_ann_file,
                                   pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(ann_file=test_ann_file,
                                    pipeline=test_pipeline))
val_evaluator = dict(ann_file=data_root + val_ann_file, metric='keypoints')
test_evaluator = dict(ann_file=data_root + test_ann_file, metric='keypoints')
