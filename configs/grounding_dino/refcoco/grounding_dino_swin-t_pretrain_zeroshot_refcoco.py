_base_ = '../grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'

model = dict(test_cfg=dict(max_per_img=15))

data_root = '/home/PJLAB/huanghaian/dataset/coco2014/'

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
        ann_file='mdetr_annotations/finetune_refcoco_val.json',
        data_prefix=dict(img='train2014/'),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline,
        backend_args=None))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='RefExpMetric',
    ann_file=data_root + 'mdetr_annotations/finetune_refcoco_val.json',
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))
test_evaluator = val_evaluator
