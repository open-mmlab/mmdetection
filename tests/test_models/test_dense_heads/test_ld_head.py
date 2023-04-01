# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.models.dense_heads import GFLHead, LDHead


def test_ld_head_loss():
    """Tests vfnet head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(type='ATSSAssigner', topk=9, ignore_iof_thr=0.1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))

    self = LDHead(
        num_classes=4,
        in_channels=1,
        train_cfg=train_cfg,
        loss_ld=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]))

    teacher_model = GFLHead(
        num_classes=4,
        in_channels=1,
        train_cfg=train_cfg,
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]))

    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    cls_scores, bbox_preds = self.forward(feat)
    rand_soft_target = teacher_model.forward(feat)[1]

    # Test that empty ground truth encourages the network to predict
    # background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None

    empty_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                rand_soft_target, img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero, ld loss should
    # be non-negative but there should be no box loss.
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_box_loss = sum(empty_gt_losses['loss_bbox'])
    empty_ld_loss = sum(empty_gt_losses['loss_ld'])
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')
    assert empty_ld_loss.item() >= 0, 'ld loss should be non-negative'

    # When truth is non-empty then both cls and box loss should be nonzero
    # for random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                              rand_soft_target, img_metas, gt_bboxes_ignore)
    onegt_cls_loss = sum(one_gt_losses['loss_cls'])
    onegt_box_loss = sum(one_gt_losses['loss_bbox'])

    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'

    gt_bboxes_ignore = gt_bboxes

    # When truth is non-empty but ignored then the cls loss should be nonzero,
    # but there should be no box loss.
    ignore_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                 rand_soft_target, img_metas, gt_bboxes_ignore)
    ignore_cls_loss = sum(ignore_gt_losses['loss_cls'])
    ignore_box_loss = sum(ignore_gt_losses['loss_bbox'])

    assert ignore_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert ignore_box_loss.item() == 0, 'gt bbox ignored loss should be zero'

    # When truth is non-empty and not ignored then both cls and box loss should
    # be nonzero for random inputs
    gt_bboxes_ignore = [torch.randn(1, 4)]

    not_ignore_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes,
                                     gt_labels, rand_soft_target, img_metas,
                                     gt_bboxes_ignore)
    not_ignore_cls_loss = sum(not_ignore_gt_losses['loss_cls'])
    not_ignore_box_loss = sum(not_ignore_gt_losses['loss_bbox'])

    assert not_ignore_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert not_ignore_box_loss.item(
    ) > 0, 'gt bbox not ignored loss should be non-zero'
