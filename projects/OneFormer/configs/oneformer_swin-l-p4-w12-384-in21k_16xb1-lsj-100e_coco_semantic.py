_base_ = ['./_base_/oneformer_swin-l-p4-w12-384-in21k_semantic.py']

model = dict(
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=True,
        instance_on=False,
    ), )

dataset_type = 'CocoSegDataset'
data_root = 'data/coco/'
backend_args = None
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='pillow',
        backend_args=backend_args),
    dict(
        type='ResizeShortestEdge',
        scale=800,
        max_size=1333,
        backend='pillow',
        interpolation='bilinear'),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_mask=False,
        with_seg=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
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
        use_label_map=False,
        data_prefix=dict(
            img_path='val2017/',
            seg_map_path='annotations/panoptic_semseg_val2017/'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = [dict(type='SemSegMetric', iou_metrics=['mIoU'])]
test_evaluator = val_evaluator

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        # imdecode_backend='pillow',
        backend_args=backend_args),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        backend_args=backend_args),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=_base_.image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=_base_.image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='PackDetInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
