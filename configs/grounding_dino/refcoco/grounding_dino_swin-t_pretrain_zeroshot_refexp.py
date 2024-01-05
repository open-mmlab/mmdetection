_base_ = '../grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'

# 30 is an empirical value, just set it to the maximum value
# without affecting the evaluation result
model = dict(test_cfg=dict(max_per_img=30))

data_root = 'data/coco/'

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
                   'tokens_positive'))
]

# -------------------------------------------------#
ann_file = 'mdetr_annotations/final_refexp_val.json'
val_dataset_all_val = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)
val_evaluator_all_val = dict(
    type='RefExpMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_refcoco_testA.json'
val_dataset_refcoco_testA = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)

val_evaluator_refcoco_testA = dict(
    type='RefExpMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_refcoco_testB.json'
val_dataset_refcoco_testB = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)

val_evaluator_refcoco_testB = dict(
    type='RefExpMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_refcoco+_testA.json'
val_dataset_refcoco_plus_testA = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)

val_evaluator_refcoco_plus_testA = dict(
    type='RefExpMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_refcoco+_testB.json'
val_dataset_refcoco_plus_testB = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)

val_evaluator_refcoco_plus_testB = dict(
    type='RefExpMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_refcocog_test.json'
val_dataset_refcocog_test = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)

val_evaluator_refcocog_test = dict(
    type='RefExpMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_grefcoco_val.json'
val_dataset_grefcoco_val = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)

val_evaluator_grefcoco_val = dict(
    type='gRefCOCOMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    thresh_score=0.7,
    thresh_f1=1.0)

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_grefcoco_testA.json'
val_dataset_grefcoco_testA = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)

val_evaluator_grefcoco_testA = dict(
    type='gRefCOCOMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    thresh_score=0.7,
    thresh_f1=1.0)

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_grefcoco_testB.json'
val_dataset_grefcoco_testB = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=test_pipeline,
    backend_args=None)

val_evaluator_grefcoco_testB = dict(
    type='gRefCOCOMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    thresh_score=0.7,
    thresh_f1=1.0)

# -------------------------------------------------#
datasets = [
    val_dataset_all_val, val_dataset_refcoco_testA, val_dataset_refcoco_testB,
    val_dataset_refcoco_plus_testA, val_dataset_refcoco_plus_testB,
    val_dataset_refcocog_test, val_dataset_grefcoco_val,
    val_dataset_grefcoco_testA, val_dataset_grefcoco_testB
]
dataset_prefixes = [
    'val', 'refcoco_testA', 'refcoco_testB', 'refcoco+_testA',
    'refcoco+_testB', 'refcocog_test', 'grefcoco_val', 'grefcoco_testA',
    'grefcoco_testB'
]
metrics = [
    val_evaluator_all_val, val_evaluator_refcoco_testA,
    val_evaluator_refcoco_testB, val_evaluator_refcoco_plus_testA,
    val_evaluator_refcoco_plus_testB, val_evaluator_refcocog_test,
    val_evaluator_grefcoco_val, val_evaluator_grefcoco_testA,
    val_evaluator_grefcoco_testB
]

val_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
