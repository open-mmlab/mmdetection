_base_ = 'fcos_r50_caffe_fpn_gn-head_1x_TinyPerson640.py'
INF = 1e8
model = dict(
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    neck=dict(
        start_level=0,   # add
    ),
    bbox_head=dict(
        norm_cfg=None,   # add
        num_classes=1,   # add
        strides=[4, 8, 16, 32, 64],  # add
        regress_ranges=((-1, 16), (16, 32), (32, 64), (64, 128), (128, INF)),  # add
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=1000)
)

# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', scale_factor=[1.0], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        # type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        type='CroppedTilesFlipAug',
        tile_shape=(640, 512),  # sub image size by cropped
        tile_overlap=(100, 100),
        scale_factor=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoFmtDataset'
data_root = 'data/tiny_set/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'mini_annotations/tiny_set_test_all.json',
        # ann_file=data_root + 'annotations/corner/task/tiny_set_test_sw640_sh512_all.json',
        # merge_after_infer_kwargs=dict(
        #     merge_gt_file=data_root + 'mini_annotations/tiny_set_test_all.json',
        #     merge_nms_th=0.5
        # ),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer_config = dict(_delete_=True, grad_clip=None)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup='linear')
