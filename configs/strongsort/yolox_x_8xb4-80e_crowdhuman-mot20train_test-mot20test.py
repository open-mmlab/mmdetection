_base_ = ['./yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py']

data_root = 'data/MOT20/'

img_scale = (1600, 896)  # width, height

# model settings
model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(type='BatchSyncRandomResize', random_size_range=(640, 1152))
    ]))

train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=True),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=True),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Resize',
        scale=img_scale,
        keep_ratio=True,
        clip_object_border=True),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='CocoDataset',
                    data_root=data_root,
                    ann_file='annotations/train_cocoformat.json',
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian', )),
                    pipeline=[
                        dict(
                            type='LoadImageFromFile',
                            backend_args=_base_.backend_args),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ]),
                dict(
                    type='CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_train.json',
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian', )),
                    pipeline=[
                        dict(
                            type='LoadImageFromFile',
                            backend_args=_base_.backend_args),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ]),
                dict(
                    type='CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_val.json',
                    data_prefix=dict(img='val'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian', )),
                    pipeline=[
                        dict(
                            type='LoadImageFromFile',
                            backend_args=_base_.backend_args),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ]),
            ]),
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        data_root='data/MOT17', ann_file='annotations/train_cocoformat.json'))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(ann_file='data/MOT17/annotations/train_cocoformat.json')
test_evaluator = val_evaluator
