_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

data_root = 'data/coco/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # change this
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file='mdetr_annotations/finetune_refcoco_train_vg.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=train_pipeline))

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_refcoco_val.json'
val_dataset_all_val = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=_base_.test_pipeline,
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
    pipeline=_base_.test_pipeline,
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
    pipeline=_base_.test_pipeline,
    backend_args=None)

val_evaluator_refcoco_testB = dict(
    type='RefExpMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

# -------------------------------------------------#
datasets = [
    val_dataset_all_val, val_dataset_refcoco_testA, val_dataset_refcoco_testB
]
dataset_prefixes = ['refcoco_val', 'refcoco_testA', 'refcoco_testB']
metrics = [
    val_evaluator_all_val, val_evaluator_refcoco_testA,
    val_evaluator_refcoco_testB
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

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            # 'language_model': dict(lr_mult=0),
        }))

# learning policy
max_epochs = 5
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
