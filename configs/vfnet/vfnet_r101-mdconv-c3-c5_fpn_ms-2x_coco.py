_base_ = './vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco.py'
model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
