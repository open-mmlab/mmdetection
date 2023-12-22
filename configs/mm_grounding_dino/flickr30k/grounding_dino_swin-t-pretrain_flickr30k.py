_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

dataset_type = 'Flickr30kDataset'
data_root = 'data/flickr30k_entities/'

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
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive', 'phrase_ids', 'phrases'))
]

dataset_Flickr30k_val = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='final_flickr_separateGT_val.json',
    data_prefix=dict(img='flickr30k_images/'),
    pipeline=test_pipeline,
)

dataset_Flickr30k_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='final_flickr_separateGT_test.json',
    data_prefix=dict(img='flickr30k_images/'),
    pipeline=test_pipeline,
)

val_evaluator_Flickr30k = dict(type='Flickr30kMetric')

test_evaluator_Flickr30k = dict(type='Flickr30kMetric')

# ----------Config---------- #
dataset_prefixes = ['Flickr30kVal', 'Flickr30kTest']
datasets = [dataset_Flickr30k_val, dataset_Flickr30k_test]
metrics = [val_evaluator_Flickr30k, test_evaluator_Flickr30k]

val_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
