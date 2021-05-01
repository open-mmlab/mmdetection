_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='open-mmlab://contrib/resnet50_gn',
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
