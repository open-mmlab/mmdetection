_base_ = './solov2-light_r50_fpn_ms-3x_coco.py'

# model settings
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    mask_head=dict(
        feat_channels=256,
        stacked_convs=3,
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        mask_feature_head=dict(out_channels=128),
        dcn_cfg=dict(type='DCNv2'),
        dcn_apply_to_all_conv=False))  # light solov2 head
