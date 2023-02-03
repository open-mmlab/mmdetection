_base_ = '../cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-1, 1),
        color_prob=0.5,
        hflag=False,
        aug_ratio=0.5),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

max_epochs = 48

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs)

# only keep latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
