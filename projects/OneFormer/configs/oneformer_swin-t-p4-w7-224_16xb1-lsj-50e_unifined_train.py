_base_ = ['./_base_/oneformer_swin-t-p4-w7-224_panoptic.py', 
          'mmdet::_base_/datasets/coco_panoptic.py']

model = dict(
    test_cfg=dict(
        panoptic_on=True,
        semantic_on=True,
        instance_on=True,
        ),
)
backend_args=None
dataset_type = 'CocoDataset'
data_root='data/coco/'
test_pipeline = [
    dict(type='LoadImageFromFile',
         imdecode_backend='pillow', 
         backend_args=backend_args),
    dict(
        type='ResizeShortestEdge', scale=800, max_size=1333, backend='pillow', interpolation='bilinear'),
    dict(type='LoadPanopticAnnotations', imdecode_backend='pillow', backend_args=backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='CocoPanopticMetric',
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        seg_prefix=data_root + 'annotations/panoptic_val2017/',
        backend_args=backend_args),
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_val2017.json',
        metric=['bbox', 'segm'],
        backend_args=backend_args),
    dict(
        type='SemSegMetric',
        iou_metrics=['mIoU'])
]
test_evaluator = val_evaluator

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        imdecode_backend='pillow',
        backend_args=backend_args),
    dict(
        type='LoadPanopticAnnotations',
        imdecode_backend='pillow', 
        with_bbox=True,  
        with_mask=True,
        with_seg=True,
        backend_args=backend_args,
        ),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
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
    dict(
        type='PreprocessAnnotationsOneFormer',
        ),
    dict(type='PackDetInputs', 
         meta_keys=('img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'oneformer_texts', 'task'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        pipeline=train_pipeline, 
        return_classes=True))
