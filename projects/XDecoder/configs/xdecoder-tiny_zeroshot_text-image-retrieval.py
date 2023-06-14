_base_ = 'xdecoder-tiny_zeroshot_caption_coco2014.py'

model = dict(head=dict(task='retrieval'))

grounding_scale = 512

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='pillow',
        backend_args=_base_.backend_args),
    dict(
        type='ResizeShortestEdge',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text'))
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
