_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/default_runtime.py',
    'semi_coco_detection.py'
]

model_wrapper = dict(
    type='SoftTeacher',
    detector='${model}',
    semi_train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        sup_weight=1.0,
        unsup_weight=1.0,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(infer_on='teacher'),
    data_preprocessor=dict(
        type='MultiDataPreprocessor',
        data_preprocessor='${model.data_preprocessor}',
    ))

# training schedule for 90k
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

custom_hooks = [dict(type='MeanTeacherHook')]

load_from = '/home/SENSETIME/chenzeming.vendor/Documents/gitlib/' \
            'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
