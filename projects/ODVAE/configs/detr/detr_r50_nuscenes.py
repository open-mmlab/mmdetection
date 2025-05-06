# detr_resnet_nuscenes.py - Specific config for DETR with ResNet backbone on NuScenes

_base_ = ['./detr_base.py']

# Update the bbox_head to use the correct number of classes
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        in_channels=[2048],  # Match ResNet50 output channels
    ),
    bbox_head=dict(
        num_classes=len(_base_.class_names),  # Use class_names from base_nuscenes.py
    )
)

# Optimizer - specific for ResNet
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}
    )
)

# Output directories
val_evaluator = dict(
    outfile_prefix='.out/work_dirs/detr_resnet_nuscenes/results',
)

# Update default hooks with specific output paths
default_hooks = dict(
    checkpoint=dict(
        out_dir='.out/work_dirs/detr_resnet_nuscenes/checkpoints',
    ),
)
