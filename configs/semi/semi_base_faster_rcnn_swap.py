_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/default_runtime.py',
    'semi_coco_detection.py'
]

model_wrapper = dict(
    type='SemiBaseDetector',
    detector='${model}',
    semi_train_cfg=dict(
        sup_weight=1.0,
        unsup_weight=2.0,
        score_thr=0.75,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(infer_on='teacher'),
    data_preprocessor=dict(
        type='MultiDataPreprocessor',
        data_preprocessor='${model.data_preprocessor}',
    ))

# training schedule for 180k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=90000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=90000,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=10000))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='ModelSwapHook')]

# load_from = 'D:/GitLab/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
