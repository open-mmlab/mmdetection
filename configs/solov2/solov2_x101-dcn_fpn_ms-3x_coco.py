_base_ = './solov2_r50_fpn_ms-3x_coco.py'

# model settings
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    mask_head=dict(
        mask_feature_head=dict(conv_cfg=dict(type='DCNv2')),
        dcn_cfg=dict(type='DCNv2'),
        dcn_apply_to_all_conv=True))
