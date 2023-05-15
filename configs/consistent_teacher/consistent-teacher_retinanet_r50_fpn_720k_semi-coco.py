_base_ = ['./consistent-teacher_retinanet_r50_fpn_180k_semi-0.1-coco.py']

model = dict(
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=False)),
    train_cfg=dict(type='DynamicSoftLabelAssigner', topk=13, iou_factor=3.0))

# full coco train2017 is set as labeled dataset
fold = 1
percent = 100
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'data/coco/annotations/instances_train2017.json'
unlabeled_dataset.ann_file = 'data/coco/annotations/instances_unlabeled2017.json'

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=720000, val_interval=4000)

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=720000,
        by_epoch=False,
        milestones=[480000, 640000],
        gamma=0.1)
]

custom_hooks = [
    dict(type='MeanTeacher', momentum=0.9998, interval=1, warm_up=0),
]
