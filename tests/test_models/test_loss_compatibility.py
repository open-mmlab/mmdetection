import numpy as np
import torch
from mmcv import ConfigDict

from mmdet.models import build_detector


def test_loss_compatibility():
    """Test loss_cls and loss_bbox compatibility.

    Using Faster R-CNN as a sample, modifying the loss function in the config
    file to verify the compatibility of Loss APIS
    """
    # Faster R-CNN config dict
    cfg_model = dict(
        type='FasterRCNN',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=3,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            out_indices=(2, ),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='caffe'),
        rpn_head=dict(
            type='RPNHead',
            in_channels=1024,
            feat_channels=1024,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[2, 4, 8, 16, 32],
                ratios=[0.5, 1.0, 2.0],
                strides=[16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            shared_head=dict(
                type='ResLayer',
                depth=50,
                stage=3,
                stride=2,
                dilation=1,
                style='caffe',
                norm_cfg=dict(type='BN', requires_grad=False),
                norm_eval=True),
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=1024,
                featmap_strides=[16]),
            bbox_head=dict(
                type='BBoxHead',
                with_avg_pool=True,
                roi_feat_size=7,
                in_channels=2048,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
        # model training and testing settings
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=12000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=6000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)))

    # bbox loss function used to verify compatibility
    loss_bboxes = [
        dict(type='L1Loss', loss_weight=1.0),
        dict(type='GHMR', mu=0.02, bins=10, momentum=0.7, loss_weight=10.0),
        dict(type='IoULoss', loss_weight=1.0),
        dict(type='BoundedIoULoss', loss_weight=1.0),
        dict(type='GIoULoss', loss_weight=1.0),
        dict(type='DIoULoss', loss_weight=1.0),
        dict(type='CIoULoss', loss_weight=1.0),
        dict(type='MSELoss', loss_weight=1.0),
        dict(type='SmoothL1Loss', loss_weight=1.0),
        dict(type='BalancedL1Loss', loss_weight=1.0)
    ]
    # class loss function used to verify compatibility
    loss_clses = [
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        dict(
            type='GHMC',
            bins=30,
            momentum=0.75,
            use_sigmoid=True,
            loss_weight=1.0)
    ]
    # set some important args in forward_train
    imgs = torch.randn(1, 3, 224, 224)
    gt_bboxes = [torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])]
    gt_labels = [torch.LongTensor([2])]
    img_matas = {
        'filename': '000000037777.jpg',
        'ori_filename': '000000037777.jpg',
        'ori_shape': (224, 224, 3),
        'img_shape': (224, 224, 3),
        'pad_shape': (224, 224, 3),
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': None,
        'img_norm_cfg': {
            'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
            'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
            'to_rgb': True
        }
    }
    img_metas = [img_matas]
    cfg_model = ConfigDict(cfg_model)

    # verify bbox loss function compatibility
    for loss_bbox in loss_bboxes:
        cfg_model.roi_head.bbox_head.loss_bbox = loss_bbox
        model = build_detector(cfg_model)
        loss = model.forward_train(imgs, img_metas, gt_bboxes, gt_labels)
        assert isinstance(loss['loss_cls'], torch.Tensor)
        assert isinstance(loss['loss_bbox'], torch.Tensor)

    # verify class loss function compatibility
    for loss_cls in loss_clses:
        cfg_model.roi_head.bbox_head.loss_cls = loss_cls
        model = build_detector(cfg_model)
        loss = model.forward_train(imgs, img_metas, gt_bboxes, gt_labels)
        assert isinstance(loss['loss_cls'], torch.Tensor)
        assert isinstance(loss['loss_bbox'], torch.Tensor)
