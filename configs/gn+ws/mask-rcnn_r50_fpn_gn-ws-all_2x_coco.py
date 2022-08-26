_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    backbone=dict(
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://jhu/resnet50_gn_ws')),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg),
        mask_head=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg)))
# learning policy
max_epochs = 24
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]
