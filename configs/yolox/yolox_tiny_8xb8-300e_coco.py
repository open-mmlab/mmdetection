_base_ = './yolox_s_8xb8-300e_coco.py'

# model settings
model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            random_size_range=(320, 640),
            size_divisor=32,
            interval=10)
    ]),
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96))

img_scale = (640, 640)  # height, width

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # Resize and Pad are for the last 15 epochs when Mosaic and
    # RandomAffine are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(416, 416), keep_ratio=True),
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
