_base_ = "faster_rcnn_r50_lka_fpn_1x_TinyPerson640_newData.py"

model = dict(
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_eval=True,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'
    ),
    neck=dict(
            type='lka_FPN',
            in_channels=[64, 128, 256, 512],
            out_channels=64,
            num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=64,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2],  # [8]
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=64,
            fc_out_channels=256,
            roi_feat_size=7,
            num_classes=1,  # 80
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # train_cfg=dict(
    #         rpn=dict(
    #             assigner=dict(
    #                 type='MaxIoUAssigner',
    #                 pos_iou_thr=0.7,
    #                 neg_iou_thr=0.3,
    #                 min_pos_iou=0.3,
    #                 match_low_quality=True,
    #                 ignore_iof_thr=-1),
    #             sampler=dict(
    #                 type='RandomSampler',
    #                 num=64,
    #                 pos_fraction=0.5,
    #                 neg_pos_ub=-1,
    #                 add_gt_as_proposals=False),
    #             allowed_border=-1,
    #             pos_weight=-1,
    #             debug=False),
    #         rpn_proposal=dict(
    #             nms_across_levels=False,
    #             nms_pre=2000,
    #             nms_post=1000,
    #             max_num=1000,
    #             nms=dict(type='nms', iou_threshold=0.7),
    #             min_bbox_size=0),
    #         rcnn=dict(
    #             assigner=dict(
    #                 type='MaxIoUAssigner',
    #                 pos_iou_thr=0.5,
    #                 neg_iou_thr=0.5,
    #                 min_pos_iou=0.5,
    #                 match_low_quality=False,
    #                 ignore_iof_thr=-1),
    #             sampler=dict(
    #                 type='RandomSampler',
    #                 num=128,
    #                 pos_fraction=0.25,
    #                 neg_pos_ub=-1,
    #                 add_gt_as_proposals=True),
    #             pos_weight=-1,
    #             debug=False)),
)