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
        pad_size_divisor=32
    ),
    backbone=dict(type='FocalNet'),
    head=dict(type='XDecoderUnifiedhead',
              in_channels=(96, 192, 384, 768),
              task='semseg',
              pixel_decoder=dict(type='TransformerEncoderPixelDecoder'),
              transformer_decoder=dict(type='XDecoderTransformerDecoder'),
              ),
    test_cfg=dict(mask_thr=0.5,
                  use_thr_for_mc=True)  # mc means multi-class
)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow', backend_args=backend_args),
    dict(type='FixScaleResize',
         scale=512,
         keep_ratio=True,
         short_side_mode=True,
         backend='pillow',
         interpolation='bicubic'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text'))
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader
