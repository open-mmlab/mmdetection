_base_ = ['../yolox/yolox_x_8xb8-300e_coco.py']

data_root = 'data/MOT17/'

img_scale = (800, 1440)  # width, height
batch_size = 4

# model settings
model = dict(
    bbox_head=dict(num_classes=1), test_cfg=dict(nms=dict(iou_threshold=0.7)))

train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Resize',
        scale=img_scale,
        keep_ratio=True,
        clip_object_border=False),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='CocoDataset',
                    data_root=data_root,
                    ann_file='annotations/half-train_cocoformat.json',
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ]),
                dict(
                    type='CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_train.json',
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ]),
                dict(
                    type='CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_val.json',
                    data_prefix=dict(img='val'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ]),
            ]),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img='train'),
        metainfo=dict(classes=('pedestrian', )),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# training settings
max_epochs = 80
num_last_epochs = 10
interval = 5

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.001 / 8 * batch_size
optim_wrapper = dict(optimizer=dict(lr=base_lr))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    visualization=dict(type='TrackVisualizationHook', draw=False))

# visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# evaluator
val_evaluator = dict(ann_file=data_root +
                     'annotations/half-val_cocoformat.json')
test_evaluator = val_evaluator

del _base_.tta_model
del _base_.tta_pipeline
del _base_.train_dataset
