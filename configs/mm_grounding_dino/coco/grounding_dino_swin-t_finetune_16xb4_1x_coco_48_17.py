_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

data_root = 'data/coco/'
base_classes = ('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck',
                'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra',
                'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee',
                'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'microwave', 'oven', 'toaster',
                'refrigerator', 'book', 'clock', 'vase', 'toothbrush')  # 48
novel_classes = ('airplane', 'bus', 'cat', 'dog', 'cow', 'elephant',
                 'umbrella', 'tie', 'snowboard', 'skateboard', 'cup', 'knife',
                 'cake', 'couch', 'keyboard', 'sink', 'scissors')  # 17
all_classes = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'kite', 'skateboard',
    'surfboard', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush')  # 65

train_metainfo = dict(classes=base_classes)
test_metainfo = dict(
    classes=all_classes,
    base_classes=base_classes,
    novel_classes=novel_classes)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        metainfo=train_metainfo,
        data_root=data_root,
        ann_file='annotations/instances_train2017_seen_2.json',
        data_prefix=dict(img='train2017/'),
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=test_metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val2017_all_2.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        return_classes=True,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='OVCocoMetric',
    ann_file=data_root + 'annotations/instances_val2017_all_2.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            # 'language_model': dict(lr_mult=0),
        }))

# learning policy
max_epochs = 12
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=1, save_best='coco/novel_ap50', rule='greater'))

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
