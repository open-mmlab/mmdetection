_base_ = [
    './fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/semi_coco_detection.py',
]
detector = _base_.model
model = dict(
    _delete_=True,
    type='DenseTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        burn_in_steps=5000,
        sup_weight=1.0,
        unsup_weight=4.0,
        k_ratio=0.01,
        logits_weight=4.0,
        deltas_weight=1.0,
        quality_weight=1.0,
    ),
    semi_test_cfg=dict(predict_on='teacher'))

unsup_pipeline = _base_.unsup_pipeline
unsup_pipeline[2] = dict(
    type='DenseMultiBranch',
    branch_field=_base_.branch_field,
    unsup_teacher=_base_.weak_pipeline,
    unsup_student=_base_.strong_pipeline)

batch_size = 48
num_workers = 16

# 10% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10.json'
unlabeled_dataset.ann_file = 'semi_anns/' \
                             'instances_train2017.1@10-unlabeled.json'
unlabeled_dataset.data_prefix = dict(img='train2017/')
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]),
    sampler=dict(batch_size=batch_size,
                 source_ratio=[2, 1]))

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=180000, val_interval=5000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=180000,
        by_epoch=False,
        milestones=[179995],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=5000, max_keep_ckpts=2))

log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook', start_steps=3000, momentum=0.0004), dict(type='SetEpochInfoHook')]
