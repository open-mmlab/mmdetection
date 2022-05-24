# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.models.dense_heads import DDODHead


def test_ddod_head_loss():
    """Tests ddod head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(  # ATSSAssigner
            assigner=dict(type='ATSSAssigner', topk=9, alpha=0.8),
            reg_assigner=dict(type='ATSSAssigner', topk=9, alpha=0.5),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    self = DDODHead(
        num_classes=4,
        in_channels=1,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        train_cfg=train_cfg,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_iou=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    cls_scores, bbox_preds, iou_preds = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, iou_preds, gt_bboxes,
                                gt_labels, img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_box_loss = sum(empty_gt_losses['loss_bbox'])
    empty_iou_loss = sum(empty_gt_losses['loss_iou'])
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')
    assert empty_iou_loss.item() == 0, (
        'there should be no iou loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, iou_preds, gt_bboxes,
                              gt_labels, img_metas, gt_bboxes_ignore)
    onegt_cls_loss = sum(one_gt_losses['loss_cls'])
    onegt_box_loss = sum(one_gt_losses['loss_bbox'])
    onegt_iou_loss = sum(one_gt_losses['loss_iou'])
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
    assert onegt_iou_loss.item() > 0, 'iou loss should be non-zero'
