_base_ = '../grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'

data_root = 'data/d3/'

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities', 'sent_ids'))
]

val_dataloader = dict(
    dataset=dict(_delete_=True,
                 type='DODDataset',
                 data_root=data_root,
                 ann_file='d3_json/d3_full_annotations.json',
                 data_prefix=dict(img='d3_images/', anno='d3_pkl'),
                 pipeline=test_pipeline,
                 test_mode=True,
                 backend_args=None,
                 return_classes=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='DODCocoMetric',
    ann_file=data_root + 'd3_json/d3_full_annotations.json')
test_evaluator = val_evaluator
