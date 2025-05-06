# detr_ldmencoder_nuscenes.py - Specific config for DETR with LDMEncoder backbone on NuScenes

_base_ = ['./detr_base.py']

# Custom imports for the LDMEncoder backbone
custom_imports = dict(
    imports=['projects.ODVAE.odvae.backbones.vae_encoder'],
    allow_failed_imports=False)

# Update model configuration for LDMEncoder backbone
model = dict(
    backbone=dict(
        type='LDMEncoderBackbone',
        in_channels=3,
        out_indices=(0,),  # Only use the final output
        pretrained='pretrained_checkpoints/kl-f16/model.ckpt',
        freeze=True,
        # Add any required parameters for LDM encoder here
    ),
    neck=dict(
        in_channels=[32],  # Match the output channels of the LDM encoder
    ),
    bbox_head=dict(
        num_classes=len(_base_.class_names),  # Use class_names from base_nuscenes.py
    )
)

# Optimizer - specific for LDMEncoder (freeze backbone)
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.0, decay_mult=1.0)}
    )
)

# Output directories
val_evaluator = dict(
    outfile_prefix='.out/work_dirs/detr_ldm_nuscenes/results',
)

# Update default hooks with specific output paths
default_hooks = dict(
    checkpoint=dict(
        out_dir='.out/work_dirs/detr_ldm_nuscenes/checkpoints',
    ),
)