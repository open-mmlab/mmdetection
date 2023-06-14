_base_ = [
    '_base_/xdecoder-tiny_open-vocab-panoptic.py',
    'mmdet::_base_/datasets/ade20k_panoptic.py'
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='pillow',
        backend_args=_base_.backend_args),
    dict(
        type='ResizeShortestEdge', scale=640, max_size=2560, backend='pillow'),
    dict(type='LoadPanopticAnnotations', backend_args=_base_.backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text'))
]

val_dataloader = dict(
    dataset=dict(return_classes=True, pipeline=test_pipeline))
test_dataloader = val_dataloader
