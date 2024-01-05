_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

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

# -------------------------------------------------#
val_dataset_full = dict(
    type='DODDataset',
    data_root=data_root,
    ann_file='d3_json/d3_full_annotations.json',
    data_prefix=dict(img='d3_images/', anno='d3_pkl'),
    pipeline=test_pipeline,
    test_mode=True,
    backend_args=None,
    return_classes=True)

val_evaluator_full = dict(
    type='DODCocoMetric',
    ann_file=data_root + 'd3_json/d3_full_annotations.json')

# -------------------------------------------------#
val_dataset_pres = dict(
    type='DODDataset',
    data_root=data_root,
    ann_file='d3_json/d3_pres_annotations.json',
    data_prefix=dict(img='d3_images/', anno='d3_pkl'),
    pipeline=test_pipeline,
    test_mode=True,
    backend_args=None,
    return_classes=True)
val_evaluator_pres = dict(
    type='DODCocoMetric',
    ann_file=data_root + 'd3_json/d3_pres_annotations.json')

# -------------------------------------------------#
val_dataset_abs = dict(
    type='DODDataset',
    data_root=data_root,
    ann_file='d3_json/d3_abs_annotations.json',
    data_prefix=dict(img='d3_images/', anno='d3_pkl'),
    pipeline=test_pipeline,
    test_mode=True,
    backend_args=None,
    return_classes=True)
val_evaluator_abs = dict(
    type='DODCocoMetric',
    ann_file=data_root + 'd3_json/d3_abs_annotations.json')

# -------------------------------------------------#
datasets = [val_dataset_full, val_dataset_pres, val_dataset_abs]
dataset_prefixes = ['FULL', 'PRES', 'ABS']
metrics = [val_evaluator_full, val_evaluator_pres, val_evaluator_abs]

val_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
