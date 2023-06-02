_base_ = [
    '../_base_/models/faster-rcnn_r50-caffe-c4.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='VOCDataset',
                data_root={{_base_.data_root}},
                ann_file='VOC2007/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2007/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
                backend_args={{_base_.backend_args}}),
            dict(
                type='VOCDataset',
                data_root={{_base_.data_root}},
                ann_file='VOC2012/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2012/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
                backend_args={{_base_.backend_args}})
        ]))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# training schedule for 18k
max_iter = 18000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=3000)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[12000, 16000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=3000))
log_processor = dict(by_epoch=False)
