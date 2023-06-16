_base_ = [
    '_base_/xdecoder-tiny_open-vocab-panoptic.py',
    'mmdet::_base_/datasets/coco_panoptic.py'
]

model = dict(test_cfg=dict(mask_thr=0.4))

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='pillow',
        backend_args=_base_.backend_args),
    dict(
        type='ResizeShortestEdge', scale=800, max_size=1333, backend='pillow'),
    dict(type='LoadPanopticAnnotations', backend_args=_base_.backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'stuff_text'))
]

val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, return_classes=True))

test_dataloader = val_dataloader

train_dataloader = None
