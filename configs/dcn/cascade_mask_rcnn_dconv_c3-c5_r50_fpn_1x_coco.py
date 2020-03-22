_base_ = '../cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
work_dir = './work_dirs/cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x'
