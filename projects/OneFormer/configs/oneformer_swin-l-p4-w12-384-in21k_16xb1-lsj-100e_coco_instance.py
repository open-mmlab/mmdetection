_base_ = [
    './_base_/oneformer_swin-l-p4-w12-384-in21k_instance.py',
    'mmdet::_base_/datasets/coco_instance.py',
]
model = dict(
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
    ), )
backend_args = None
data_root = 'data/coco/'

test_pipeline = [
    dict(
        type='LoadImageFromFile', imdecode_backend='pillow',
        backend_args=None),
    dict(
        type='ResizeShortestEdge', scale=800, max_size=1333, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=False, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_dataloader = dict(
    batch_size=1, num_workers=2, dataset=dict(pipeline=test_pipeline, ))
test_dataloader = val_dataloader
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/panoptic2instances_val2017.json',
        metric='segm',
        backend_args=None)
]
test_evaluator = val_evaluator
