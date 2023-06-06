_base_ = 'mmdet::_base_/default_runtime.py'

custom_imports = dict(
    imports=['projects.XDecoder.xdecoder'], allow_failed_imports=False)

model = dict(
    type='XDecoder',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(type='FocalNet'),
    head=dict(
        type='XDecoderUnifiedhead',
        in_channels=(96, 192, 384, 768),
        pixel_decoder=dict(type='XTransformerEncoderPixelDecoder'),
        transformer_decoder=dict(type='XDecoderTransformerDecoder'),
        task='semseg',
    ),
    test_cfg=dict(mask_thr=0.5, use_thr_for_mc=False)  # mc means multi-class
)

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=640,
        keep_ratio=True,
        short_side_mode=True,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='LoadSemSegAnnotations'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'seg_map_path', 'img',
                   'gt_seg_map', 'text'))
]

dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        return_classes=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='SemSegMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
