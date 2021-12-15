_base_ = './tood_x101_64x4d_fpn_mstrain_2x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    bbox_head=dict(num_dcn=2))
