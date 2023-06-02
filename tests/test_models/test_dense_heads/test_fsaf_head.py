# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.models.dense_heads import FSAFHead


def test_fsaf_head_loss():
    """Tests anchor head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cfg = dict(
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(type='TBLRBBoxCoder', normalizer=4.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='none'),
        loss_bbox=dict(
            type='IoULoss', eps=1e-6, loss_weight=1.0, reduction='none'))

    train_cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='CenterRegionAssigner',
                pos_scale=0.2,
                neg_scale=0.2,
                min_pos_iof=0.01),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    head = FSAFHead(num_classes=4, in_channels=1, train_cfg=train_cfg, **cfg)
    if torch.cuda.is_available():
        head.cuda()
        # FSAF head expects a multiple levels of features per image
        feat = [
            torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2))).cuda()
            for i in range(len(head.anchor_generator.strides))
        ]
        cls_scores, bbox_preds = head.forward(feat)
        gt_bboxes_ignore = None

        # When truth is non-empty then both cls and box loss should be nonzero
        #  for random inputs
        gt_bboxes = [
            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
        ]
        gt_labels = [torch.LongTensor([2]).cuda()]
        one_gt_losses = head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                  img_metas, gt_bboxes_ignore)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert onegt_box_loss.item() > 0, 'box loss should be non-zero'

        # Test that empty ground truth encourages the network to predict bkg
        gt_bboxes = [torch.empty((0, 4)).cuda()]
        gt_labels = [torch.LongTensor([]).cuda()]

        empty_gt_losses = head.loss(cls_scores, bbox_preds, gt_bboxes,
                                    gt_labels, img_metas, gt_bboxes_ignore)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert empty_box_loss.item() == 0, (
            'there should be no box loss when there are no true boxes')
