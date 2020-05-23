import mmcv
import torch

from mmdet.models.dense_heads import PISARetinaHead, PISASSDHead
from mmdet.models.roi_heads import PISARoIHead


def test_pisa_retinanet_head_loss():
    """
    Tests pisa retinanet head loss when truth is empty and non-empty
    """
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cfg = mmcv.Config(
        dict(
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
            isr=dict(k=2., bias=0.),
            carl=dict(k=1., bias=0.2),
            allowed_border=0,
            pos_weight=-1,
            debug=False))
    self = PISARetinaHead(num_classes=4, in_channels=1, train_cfg=cfg)

    # Anchor head expects a multiple levels of features per image
    feat = [
        torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
        for i in range(len(self.anchor_generator.strides))
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
    empty_cls_loss = empty_gt_losses['loss_cls'].sum()
    empty_box_loss = empty_gt_losses['loss_bbox'].sum()
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
    onegt_cls_loss = one_gt_losses['loss_cls'].sum()
    onegt_box_loss = one_gt_losses['loss_bbox'].sum()
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'


def test_pisa_ssd_head_loss():
    """
    Tests pisa ssd head loss when truth is empty and non-empty
    """
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.,
                ignore_iof_thr=-1,
                gt_max_assign_all=False),
            isr=dict(k=2., bias=0.),
            carl=dict(k=1., bias=0.2),
            smoothl1_beta=1.,
            allowed_border=-1,
            pos_weight=-1,
            neg_pos_ratio=3,
            debug=False))
    ssd_anchor_generator = dict(
        type='SSDAnchorGenerator',
        scale_major=False,
        input_size=300,
        strides=[1],
        ratios=([2], ),
        basesize_ratio_range=(0.15, 0.9))
    self = PISASSDHead(
        num_classes=4,
        in_channels=(1, ),
        train_cfg=cfg,
        anchor_generator=ssd_anchor_generator)

    # Anchor head expects a multiple levels of features per image
    feat = [
        torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
        for i in range(len(self.anchor_generator.strides))
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
    # SSD is special, #pos:#neg = 1: 3, so empth gt will also lead loss cls = 0
    assert empty_cls_loss.item() == 0, 'cls loss should be non-zero'
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


def test_pisa_roi_head_loss():
    """
    Tests pisa roi head loss when truth is empty and non-empty
    """
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='ScoreHLRSampler',
                num=4,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
                k=0.5,
                bias=0.),
            isr=dict(k=2., bias=0.),
            carl=dict(k=1., bias=0.2),
            allowed_border=0,
            pos_weight=-1,
            debug=False))

    bbox_roi_extractor = dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=0),
        out_channels=1,
        featmap_strides=[1])

    bbox_head = dict(
        type='Shared2FCBBoxHead',
        in_channels=1,
        fc_out_channels=2,
        roi_feat_size=7,
        num_classes=4,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0))

    self = PISARoIHead(bbox_roi_extractor, bbox_head, train_cfg=train_cfg)

    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    # Anchor head expects a multiple levels of features per image
    feat = [
        torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
        for i in range(1)
    ]

    proposal_list = [
        torch.Tensor([[22.6667, 22.8757, 238.6326, 151.8874], [0, 3, 5, 7]])
    ]

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None

    empty_gt_losses = self.forward_train(feat, img_metas, proposal_list,
                                         gt_bboxes, gt_labels,
                                         gt_bboxes_ignore)

    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = empty_gt_losses['loss_cls'].sum()
    empty_box_loss = empty_gt_losses['loss_bbox'].sum()
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]

    one_gt_losses = self.forward_train(feat, img_metas, proposal_list,
                                       gt_bboxes, gt_labels, gt_bboxes_ignore)
    onegt_cls_loss = one_gt_losses['loss_cls'].sum()
    onegt_box_loss = one_gt_losses['loss_bbox'].sum()
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
