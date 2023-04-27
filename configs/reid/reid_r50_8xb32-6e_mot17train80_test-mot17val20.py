_base_ = [
    '../_base_/datasets/mot_challenge_reid.py', '../_base_/default_runtime.py'
]
model = dict(
    type='BaseReID',
    data_preprocessor=dict(
        type='ReIDDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    backbone=dict(
        type='mmcls.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
    head=dict(
        type='LinearReIDHead',
        num_fcs=1,
        in_channels=2048,
        fc_channels=1024,
        out_channels=128,
        num_classes=380,
        loss_cls=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU')),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth'  # noqa: E501
    ))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=None,
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=6,
        by_epoch=True,
        milestones=[5],
        gamma=0.1)
]

# train, val, test setting
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
