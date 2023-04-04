# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.models.dense_heads import (AscendAnchorHead, AscendRetinaHead,
                                      AscendSSDHead)


def test_ascend_anchor_head_loss():
    """Tests AscendAnchorHead loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='AscendMaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    self = AscendAnchorHead(num_classes=4, in_channels=1, train_cfg=cfg)

    # Anchor head expects a multiple levels of features per image
    feat = [
        torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
        for i in range(len(self.prior_generator.strides))
    ]
    cls_scores, bbox_preds = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_box_loss = sum(empty_gt_losses['loss_bbox'])
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                              img_metas, gt_bboxes_ignore)
    onegt_cls_loss = sum(one_gt_losses['loss_cls'])
    onegt_box_loss = sum(one_gt_losses['loss_bbox'])
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'


def test_ascend_retina_head_loss():
    """Tests AscendRetinaHead loss when truth is empty and non-empty."""
    img_shape = (800, 1067, 3)
    pad_shape = (800, 1088, 3)
    num_classes = 80
    in_channels = 256

    img_metas = [{
        'img_shape': img_shape,
        'scale_factor': 1,
        'pad_shape': pad_shape
    }]

    cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='AscendMaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    self = AscendRetinaHead(
        num_classes=num_classes, in_channels=in_channels, train_cfg=cfg)

    # Anchor head expects a multiple levels of features per image
    feat = [
        torch.rand(1, in_channels, pad_shape[0] // strides[0],
                   pad_shape[1] // strides[1])
        for strides in self.prior_generator.strides
    ]
    cls_scores, bbox_preds = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_box_loss = sum(empty_gt_losses['loss_bbox'])
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                              img_metas, gt_bboxes_ignore)
    onegt_cls_loss = sum(one_gt_losses['loss_cls'])
    onegt_box_loss = sum(one_gt_losses['loss_bbox'])
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'


def test_ascend_ssd_head_loss():
    """Tests anchor head loss when truth is empty and non-empty."""
    img_shape = (320, 320, 3)
    pad_shape = (320, 320, 3)
    in_channels = (96, 1280, 512, 256, 256, 128)
    img_metas = [{
        'img_shape': img_shape,
        'scale_factor': 1,
        'pad_shape': pad_shape
    }, {
        'img_shape': img_shape,
        'scale_factor': 1,
        'pad_shape': pad_shape
    }]

    self = AscendSSDHead(
        in_channels=in_channels,
        num_classes=80,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            strides=[16, 32, 64, 107, 160, 320],
            ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
            min_sizes=[48, 100, 150, 202, 253, 304],
            max_sizes=[100, 150, 202, 253, 304, 320]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        train_cfg=mmcv.Config(
            dict(
                assigner=dict(
                    type='AscendMaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.,
                    ignore_iof_thr=-1,
                    gt_max_assign_all=False),
                smoothl1_beta=1.,
                allowed_border=-1,
                pos_weight=-1,
                neg_pos_ratio=3,
                debug=False)))

    # Anchor head expects a multiple levels of features per image
    feat = [
        torch.rand(2, in_channels[i],
                   round(pad_shape[0] / self.prior_generator.strides[i][0]),
                   round(pad_shape[1] / self.prior_generator.strides[i][1]))
        for i in range(len(self.prior_generator.strides))
    ]
    cls_scores, bbox_preds = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4)), torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([]), torch.LongTensor([])]

    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_box_loss = sum(empty_gt_losses['loss_bbox'])
    assert empty_cls_loss.item() >= 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2]), torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                              img_metas, gt_bboxes_ignore)
    onegt_cls_loss = sum(one_gt_losses['loss_cls'])
    onegt_box_loss = sum(one_gt_losses['loss_bbox'])
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
