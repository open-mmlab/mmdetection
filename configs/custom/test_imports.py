# Simple config to test if imports work correctly
custom_imports = dict(
    imports=['mmdet.models.backbones.ldm_encoder_backbone'],
    allow_failed_imports=False)

# Just need enough to test imports
model = dict(
    type='DETR',
    backbone=dict(
        type='LDMEncoderBackbone',
        in_channels=3,
        out_indices=(0,),
        pretrained=None,
    )
)
