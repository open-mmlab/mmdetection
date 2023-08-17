_base_ = 'xdecoder-tiny_zeroshot_caption_coco2014.py'

model = dict(head=dict(task='ref-caption'))

grounding_scale = 512

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(type='ResizeShortestEdge', scale=224, backend='pillow'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text'))
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
