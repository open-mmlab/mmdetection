# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.models.dense_heads import YOLACTHead, YOLACTProtonet, YOLACTSegmHead


def test_yolact_head_loss():
    """Tests yolact head losses when truth is empty and non-empty."""
    s = 550
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0.,
                ignore_iof_thr=-1,
                gt_max_assign_all=False),
            smoothl1_beta=1.,
            allowed_border=-1,
            pos_weight=-1,
            neg_pos_ratio=3,
            debug=False,
            min_gt_box_wh=[4.0, 4.0]))
    bbox_head = YOLACTHead(
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=3,
            scales_per_octave=1,
            base_sizes=[8, 16, 32, 64, 128],
            ratios=[0.5, 1.0, 2.0],
            strides=[550.0 / x for x in [69, 35, 18, 9, 5]],
            centers=[(550 * 0.5 / x, 550 * 0.5 / x)
                     for x in [69, 35, 18, 9, 5]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='none',
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
        num_head_convs=1,
        num_protos=32,
        use_ohem=True,
        train_cfg=train_cfg)
    segm_head = YOLACTSegmHead(
        in_channels=256,
        num_classes=80,
        loss_segm=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
    mask_head = YOLACTProtonet(
        num_classes=80,
        in_channels=256,
        num_protos=32,
        max_masks_to_train=100,
        loss_mask_weight=6.125)
    feat = [
        torch.rand(1, 256, feat_size, feat_size)
        for feat_size in [69, 35, 18, 9, 5]
    ]
    cls_score, bbox_pred, coeff_pred = bbox_head.forward(feat)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_masks = [torch.empty((0, 550, 550))]
    gt_bboxes_ignore = None
    empty_gt_losses, sampling_results = bbox_head.loss(
        cls_score,
        bbox_pred,
        gt_bboxes,
        gt_labels,
        img_metas,
        gt_bboxes_ignore=gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_box_loss = sum(empty_gt_losses['loss_bbox'])
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # Test segm head and mask head
    segm_head_outs = segm_head(feat[0])
    empty_segm_loss = segm_head.loss(segm_head_outs, gt_masks, gt_labels)
    mask_pred = mask_head(feat[0], coeff_pred, gt_bboxes, img_metas,
                          sampling_results)
    empty_mask_loss = mask_head.loss(mask_pred, gt_masks, gt_bboxes, img_metas,
                                     sampling_results)
    # When there is no truth, the segm and mask loss should be zero.
    empty_segm_loss = sum(empty_segm_loss['loss_segm'])
    empty_mask_loss = sum(empty_mask_loss['loss_mask'])
    assert empty_segm_loss.item() == 0, (
        'there should be no segm loss when there are no true boxes')
    assert empty_mask_loss == 0, (
        'there should be no mask loss when there are no true boxes')

    # When truth is non-empty then cls, box, mask, segm loss should be
    # nonzero for random inputs.
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    gt_masks = [(torch.rand((1, 550, 550)) > 0.5).float()]

    one_gt_losses, sampling_results = bbox_head.loss(
        cls_score,
        bbox_pred,
        gt_bboxes,
        gt_labels,
        img_metas,
        gt_bboxes_ignore=gt_bboxes_ignore)
    one_gt_cls_loss = sum(one_gt_losses['loss_cls'])
    one_gt_box_loss = sum(one_gt_losses['loss_bbox'])
    assert one_gt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert one_gt_box_loss.item() > 0, 'box loss should be non-zero'

    one_gt_segm_loss = segm_head.loss(segm_head_outs, gt_masks, gt_labels)
    mask_pred = mask_head(feat[0], coeff_pred, gt_bboxes, img_metas,
                          sampling_results)
    one_gt_mask_loss = mask_head.loss(mask_pred, gt_masks, gt_bboxes,
                                      img_metas, sampling_results)
    one_gt_segm_loss = sum(one_gt_segm_loss['loss_segm'])
    one_gt_mask_loss = sum(one_gt_mask_loss['loss_mask'])
    assert one_gt_segm_loss.item() > 0, 'segm loss should be non-zero'
    assert one_gt_mask_loss.item() > 0, 'mask loss should be non-zero'
