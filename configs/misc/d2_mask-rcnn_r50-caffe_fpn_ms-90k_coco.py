_base_ = '../common/ms-poly-90k_coco-instance.py'

# model settings
model = dict(
    type='Detectron2Wrapper',
    bgr_to_rgb=False,
    detector=dict(
        # The settings in `d2_detector` will merged into default settings
        # in detectron2. More details please refer to
        # https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py    # noqa
        meta_architecture='GeneralizedRCNN',
        # If you want to finetune the detector, you can use the
        # checkpoint released by detectron2, for example:
        # weights='detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl'  # noqa
        weights='detectron2://ImageNetPretrained/MSRA/R-50.pkl',
        mask_on=True,
        pixel_mean=[103.530, 116.280, 123.675],
        pixel_std=[1.0, 1.0, 1.0],
        backbone=dict(name='build_resnet_fpn_backbone', freeze_at=2),
        resnets=dict(
            depth=50,
            out_features=['res2', 'res3', 'res4', 'res5'],
            num_groups=1,
            norm='FrozenBN'),
        fpn=dict(
            in_features=['res2', 'res3', 'res4', 'res5'], out_channels=256),
        anchor_generator=dict(
            name='DefaultAnchorGenerator',
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[[0.5, 1.0, 2.0]],
            angles=[[-90, 0, 90]]),
        proposal_generator=dict(name='RPN'),
        rpn=dict(
            head_name='StandardRPNHead',
            in_features=['p2', 'p3', 'p4', 'p5', 'p6'],
            iou_thresholds=[0.3, 0.7],
            iou_labels=[0, -1, 1],
            batch_size_per_image=256,
            positive_fraction=0.5,
            bbox_reg_loss_type='smooth_l1',
            bbox_reg_loss_weight=1.0,
            bbox_reg_weights=(1.0, 1.0, 1.0, 1.0),
            smooth_l1_beta=0.0,
            loss_weight=1.0,
            boundary_thresh=-1,
            pre_nms_topk_train=2000,
            post_nms_topk_train=1000,
            pre_nms_topk_test=1000,
            post_nms_topk_test=1000,
            nms_thresh=0.7,
            conv_dims=[-1]),
        roi_heads=dict(
            name='StandardROIHeads',
            num_classes=80,
            in_features=['p2', 'p3', 'p4', 'p5'],
            iou_thresholds=[0.5],
            iou_labels=[0, 1],
            batch_size_per_image=512,
            positive_fraction=0.25,
            score_thresh_test=0.05,
            nms_thresh_test=0.5,
            proposal_append_gt=True),
        roi_box_head=dict(
            name='FastRCNNConvFCHead',
            num_fc=2,
            fc_dim=1024,
            conv_dim=256,
            pooler_type='ROIAlignV2',
            pooler_resolution=7,
            pooler_sampling_ratio=0,
            bbox_reg_loss_type='smooth_l1',
            bbox_reg_loss_weight=1.0,
            bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
            smooth_l1_beta=0.0,
            cls_agnostic_bbox_reg=False),
        roi_mask_head=dict(
            name='MaskRCNNConvUpsampleHead',
            conv_dim=256,
            num_conv=4,
            pooler_type='ROIAlignV2',
            pooler_resolution=14,
            pooler_sampling_ratio=0,
            cls_agnostic_mask=False)))
