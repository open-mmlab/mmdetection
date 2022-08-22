_base_ = './yolov3_d53_mstrain-608_273e_coco.py'
# dataset settings
IM_SIZE = 320
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='BNToFP32')]
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', channel_order='rgb'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(IM_SIZE, IM_SIZE), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # image norm will be placed in IPU, and PhotoMetricDistortion has been
    # removed in this config because it has no effect on loss convergence.
    dict(type='Pad', size=(IM_SIZE, IM_SIZE)),
    dict(
        type='IPUFormatBundle',
        img_to_float=False,
        pad_dic=dict(
            gt_bboxes=dict(dim=0, shape=96),
            gt_labels=dict(dim=0, shape=96),
            gt_bboxes_ignore=dict(dim=0, shape=20))),
    dict(
        type='IPUCollect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_on=True),
    dict(
        type='GetTargetsOutsideForYolo',
        featmap_sizes=[IM_SIZE // 32, IM_SIZE // 16, IM_SIZE // 8])
]
test_pipeline = [
    dict(type='LoadImageFromFile', channel_order='rgb'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(IM_SIZE, IM_SIZE), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=(IM_SIZE, IM_SIZE), pad_val={'img': 128}),
            dict(type='IPUFormatBundle', img_to_float=False),
            dict(
                type='IPUCollect',
                keys=['img'],
                meta_tensor_keys=('scale_factor', ),
                meta_on=True)
        ])
]
data = dict(
    samples_per_gpu=48,
    train_dataloader=dict(drop_last=True, persistent_workers=True),
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

model = dict(
    backbone=dict(
        type='IPUDarknet',
        serial_num=6,
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    bbox_head=dict(static=True))

# ipu settings

options_cfg = dict(
    partialsType='half',
    enableExecutableCaching='engine_cache',
    train_cfg=dict(
        executionStrategy='SameAsIpu',
        Training=dict(gradientAccumulation=8),
        availableMemoryProportion=[0.1, 0.1, 0.1, 0.1],
    ),
    eval_cfg=dict(deviceIterations=1),
)

ipu_model_cfg = dict(split_edges=[
    dict(layer_to_call='backbone.identity', ipu_id=1),
    dict(layer_to_call='neck', ipu_id=2),
    dict(layer_to_call='bbox_head.convs_bridge.0', ipu_id=3),
])

runner = dict(
    type='IPUEpochBasedRunner',
    ipu_model_cfg=ipu_model_cfg,
    options_cfg=options_cfg,
    img_norm_cfg=img_norm_cfg,
    max_grad_norm=35)

fp16 = dict(loss_scale=512.0, velocity_accum_type='half')
