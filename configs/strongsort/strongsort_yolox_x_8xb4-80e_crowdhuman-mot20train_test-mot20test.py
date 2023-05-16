_base_ = [
    './strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain'
    '_test-mot17halfval.py'
]

img_scale = (1600, 896)  # width, height

model = dict(
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(type='BatchSyncRandomResize', random_size_range=(640, 1152))
        ]))

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
            dict(type='Resize', scale=img_scale, keep_ratio=True),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadTrackAnnotations'),
        ]),
    dict(type='PackTrackInputs')
]

val_dataloader = dict(
    dataset=dict(
        data_root='data/MOT17',
        ann_file='annotations/train_cocoformat.json',
        data_prefix=dict(img_path='train'),
        pipeline=test_pipeline))
test_dataloader = dict(
    dataset=dict(
        data_root='data/MOT20',
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img_path='test'),
        pipeline=test_pipeline))

test_evaluator = dict(format_only=True, outfile_prefix='./mot_20_test_res')
