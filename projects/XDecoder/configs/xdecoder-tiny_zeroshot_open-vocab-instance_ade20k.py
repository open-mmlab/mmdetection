_base_ = [
    '_base_/xdecoder-tiny_open-vocab-instance.py',
    'mmdet::_base_/datasets/ade20k_instance.py'
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text'))
]

val_dataloader = dict(
    dataset=dict(return_classes=True, pipeline=test_pipeline))
test_dataloader = val_dataloader

test_evaluator = dict(metric=['segm'])
