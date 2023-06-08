_base_ = [
    '_base_/xdecoder-tiny_open-vocab-semseg.py',
    'mmdet::_base_/datasets/coco_semantic.py'
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_label=False,
        with_seg=True,
        seg_reduce_indexes=[0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91],
        ignore_index=255),
    dict(
        type='PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text'))
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline, return_classes=True))
test_dataloader = val_dataloader

train_dataloader = None
