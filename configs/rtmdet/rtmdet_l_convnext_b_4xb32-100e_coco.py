_base_ = './rtmdet_l_8xb32-300e_coco.py'

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)

norm_cfg = dict(type='GN', num_groups=32)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_in1k-384px_20221219-4570f792.pth'  # noqa
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        _delete_=True,
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        batch_augments=None),
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[1, 2, 3],
        drop_path_rate=0.7,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        with_cp=True,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[256, 512, 1024], norm_cfg=norm_cfg),
    bbox_head=dict(norm_cfg=norm_cfg))

max_epochs = 100
stage2_num_epochs = 10
interval = 10
base_lr = 0.001

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

optim_wrapper = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.8,
        'decay_type': 'layer_wise',
        'num_layers': 12
    },
    optimizer=dict(lr=base_lr))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 50 to 100 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline={{_base_.train_pipeline_stage2}})
]
