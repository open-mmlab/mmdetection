_base_ = './solov2_r50_fpn_1x_coco.py'

# model settings
model = dict(
    mask_head=dict(
        stacked_convs=2,
        feat_channels=256,
        scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
        mask_feature_head=dict(out_channels=128)))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(768, 512), (768, 480), (768, 448), (768, 416), (768, 384),
                (768, 352)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(448, 768), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# training schedule for 3x
max_epochs = 36
train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]
