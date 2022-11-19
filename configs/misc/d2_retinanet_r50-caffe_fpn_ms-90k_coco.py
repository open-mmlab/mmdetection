_base_ = '../common/ms-90k_coco.py'

# model settings
model = dict(
    type='Detectron2Wrapper',
    bgr_to_rgb=False,
    detector=dict(
        # The settings in `d2_detector` will merged into default settings
        # in detectron2. More details please refer to
        # https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py    # noqa
        meta_architecture='RetinaNet',
        # If you want to finetune the detector, you can use the
        # checkpoint released by detectron2, for example:
        # weights='detectron2://COCO-Detection/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl'     # noqa
        weights='detectron2://ImageNetPretrained/MSRA/R-50.pkl',
        mask_on=False,
        pixel_mean=[103.530, 116.280, 123.675],
        pixel_std=[1.0, 1.0, 1.0],
        backbone=dict(name='build_retinanet_resnet_fpn_backbone', freeze_at=2),
        resnets=dict(
            depth=50,
            out_features=['res3', 'res4', 'res5'],
            num_groups=1,
            norm='FrozenBN'),
        fpn=dict(in_features=['res3', 'res4', 'res5'], out_channels=256),
        anchor_generator=dict(
            name='DefaultAnchorGenerator',
            sizes=[[x, x * 2**(1.0 / 3), x * 2**(2.0 / 3)]
                   for x in [32, 64, 128, 256, 512]],
            aspect_ratios=[[0.5, 1.0, 2.0]],
            angles=[[-90, 0, 90]]),
        retinanet=dict(
            num_classes=80,
            in_features=['p3', 'p4', 'p5', 'p6', 'p7'],
            num_convs=4,
            iou_thresholds=[0.4, 0.5],
            iou_labels=[0, -1, 1],
            bbox_reg_weights=(1.0, 1.0, 1.0, 1.0),
            bbox_reg_loss_type='smooth_l1',
            smooth_l1_loss_beta=0.0,
            focal_loss_gamma=2.0,
            focal_loss_alpha=0.25,
            prior_prob=0.01,
            score_thresh_test=0.05,
            topk_candidates_test=1000,
            nms_thresh_test=0.5)))

optim_wrapper = dict(optimizer=dict(lr=0.01))
