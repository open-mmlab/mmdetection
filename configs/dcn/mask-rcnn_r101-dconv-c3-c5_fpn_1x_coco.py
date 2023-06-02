_base_ = '../mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
