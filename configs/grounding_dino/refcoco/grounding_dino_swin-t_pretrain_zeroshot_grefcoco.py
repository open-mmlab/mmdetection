_base_ = '../grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'

data_root = 'data/coco2014/'
ann_file = 'mdetr_annotations/finetune_grefcoco_val.json'

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
                   'scale_factor', 'text', 'custom_entities', 'tokens_positive'))
]

val_dataloader = dict(
    dataset=dict(
        type='MDETRStyleRefCocoDataset',
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img='train2014/'),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline,
        backend_args=None))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='gRefCOCOMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    thresh_score=0.7,
    thresh_f1=1.0,
)
test_evaluator = val_evaluator
