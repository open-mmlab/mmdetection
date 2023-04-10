_base_ = ['../lvis/mask-rcnn_r50_fpn_sample1e-3_ms-1x_lvis-v1.py']

data_root = 'data/lvis/'

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="EQLV2Loss"))))

test_cfg = dict(
    rcnn=dict(
        score_thr=0.0001,
        # LVIS allows up to 300
        max_per_img=300))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            data_root=data_root,
            ann_file='annotations/lvis_v1_train.json',
            img_prefix='',
            pipeline=train_pipeline,
            upsample_thr=0.0,
        )
    ),
    val=dict(
        data_root=data_root,
        ann_file='annotations/lvis_v1_val.json',
        img_prefix='',
    ),
    test=dict(
        data_root=data_root,
        ann_file='annotations/lvis_v1_val.json',
        img_prefix='',
    )
)

evaluation = dict(interval=10, metric=['bbox', 'segm'])

test_cfg = dict(rcnn=dict(perclass_nms=True))

visualizer = dict(vis_backends=[dict(
    type='WandbVisBackend',
    init_kwargs=dict(
        project='MMDet3-EQLV2',
        name='{{fileBasenameNoExtension}}'
    )
)])