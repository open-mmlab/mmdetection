_base_ = '../deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py'  # noqa

model = dict(
    bbox_head=dict(num_classes=13204),
    test_cfg=dict(max_per_img=300),
)

data_root = 'data/V3Det/'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
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
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type='V3DetDataset',
            data_root=data_root,
            ann_file='annotations/v3det_2023_v1_train.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=False),
            pipeline=train_pipeline,
            backend_args=None)))
val_dataloader = dict(
    dataset=dict(
        type='V3DetDataset',
        data_root=data_root,
        ann_file='annotations/v3det_2023_v1_val.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/v3det_2023_v1_val.json',
    use_mp_eval=True,
    proposal_nums=[300])
test_evaluator = val_evaluator

# training schedule for 50e
# when using RFS, bs32, each epoch ~ 5730 iter
max_iter = 286500
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=max_iter / 5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[229200],  # 40e
        gamma=0.1)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5730,
        max_keep_ckpts=3))

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
