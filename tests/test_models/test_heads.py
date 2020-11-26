import mmcv
import numpy as np
import torch

from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.models.dense_heads import (AnchorHead, CornerHead, FCOSHead,
                                      FSAFHead, GuidedAnchorHead, PAAHead,
                                      SABLRetinaHead, VFNetHead, YOLACTHead,
                                      YOLACTProtonet, YOLACTSegmHead, paa_head)
from mmdet.models.dense_heads.paa_head import levels_to_images
from mmdet.models.roi_heads.bbox_heads import BBoxHead, SABLHead
from mmdet.models.roi_heads.mask_heads import FCNMaskHead, MaskIoUHead


def test_paa_head_loss():
    """Tests paa head loss when truth is empty and non-empty."""

    class mock_skm(object):

        def GaussianMixture(self, *args, **kwargs):
            return self

        def fit(self, loss):
            pass

        def predict(self, loss):
            components = np.zeros_like(loss, dtype=np.long)
            return components.reshape(-1)

        def score_samples(self, loss):
            scores = np.random.random(len(loss))
            return scores

    paa_head.skm = mock_skm()

    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.1,
                neg_iou_thr=0.1,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    # since Focal Loss is not supported on CPU
    self = PAAHead(
        num_classes=4,
        in_channels=1,
        train_cfg=train_cfg,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5))
    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    self.init_weights()
    cls_scores, bbox_preds, iou_preds = self(feat)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, iou_preds, gt_bboxes,
                                gt_labels, img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = empty_gt_losses['loss_cls']
    empty_box_loss = empty_gt_losses['loss_bbox']
    empty_iou_loss = empty_gt_losses['loss_iou']
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')
    assert empty_iou_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, iou_preds, gt_bboxes,
                              gt_labels, img_metas, gt_bboxes_ignore)
    onegt_cls_loss = one_gt_losses['loss_cls']
    onegt_box_loss = one_gt_losses['loss_bbox']
    onegt_iou_loss = one_gt_losses['loss_iou']
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
    assert onegt_iou_loss.item() > 0, 'box loss should be non-zero'
    n, c, h, w = 10, 4, 20, 20
    mlvl_tensor = [torch.ones(n, c, h, w) for i in range(5)]
    results = levels_to_images(mlvl_tensor)
    assert len(results) == n
    assert results[0].size() == (h * w * 5, c)
    assert self.with_score_voting
    cls_scores = [torch.ones(4, 5, 5)]
    bbox_preds = [torch.ones(4, 5, 5)]
    iou_preds = [torch.ones(1, 5, 5)]
    mlvl_anchors = [torch.ones(5 * 5, 4)]
    img_shape = None
    scale_factor = [0.5, 0.5]
    cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100))
    rescale = False
    self._get_bboxes_single(
        cls_scores,
        bbox_preds,
        iou_preds,
        mlvl_anchors,
        img_shape,
        scale_factor,
        cfg,
        rescale=rescale)


def test_fcos_head_loss():
    """Tests fcos head loss when truth is empty and non-empty."""
    s = 256
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
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    # since Focal Loss is not supported on CPU
    self = FCOSHead(
        num_classes=4,
        in_channels=1,
        train_cfg=train_cfg,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    cls_scores, bbox_preds, centerness = self.forward(feat)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, centerness, gt_bboxes,
                                gt_labels, img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = empty_gt_losses['loss_cls']
    empty_box_loss = empty_gt_losses['loss_bbox']
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, centerness, gt_bboxes,
                              gt_labels, img_metas, gt_bboxes_ignore)
    onegt_cls_loss = one_gt_losses['loss_cls']
    onegt_box_loss = one_gt_losses['loss_bbox']
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'


def test_vfnet_head_loss():
    """Tests vfnet head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    # since Focal Loss is not supported on CPU
    self = VFNetHead(
        num_classes=4,
        in_channels=1,
        train_cfg=train_cfg,
        loss_cls=dict(type='VarifocalLoss', use_sigmoid=True, loss_weight=1.0))
    if torch.cuda.is_available():
        self.cuda()
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size).cuda()
            for feat_size in [4, 8, 16, 32, 64]
        ]
        cls_scores, bbox_preds, bbox_preds_refine = self.forward(feat)
        # Test that empty ground truth encourages the network to predict
        # background
        gt_bboxes = [torch.empty((0, 4)).cuda()]
        gt_labels = [torch.LongTensor([]).cuda()]
        gt_bboxes_ignore = None
        empty_gt_losses = self.loss(cls_scores, bbox_preds, bbox_preds_refine,
                                    gt_bboxes, gt_labels, img_metas,
                                    gt_bboxes_ignore)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
        assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert empty_box_loss.item() == 0, (
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_bboxes = [
            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
        ]
        gt_labels = [torch.LongTensor([2]).cuda()]
        one_gt_losses = self.loss(cls_scores, bbox_preds, bbox_preds_refine,
                                  gt_bboxes, gt_labels, img_metas,
                                  gt_bboxes_ignore)
        onegt_cls_loss = one_gt_losses['loss_cls']
        onegt_box_loss = one_gt_losses['loss_bbox']
        assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert onegt_box_loss.item() > 0, 'box loss should be non-zero'


def test_anchor_head_loss():
    """Tests anchor head loss when truth is empty and non-empty."""
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
            allowed_border=0,
            pos_weight=-1,
            debug=False))
    self = AnchorHead(num_classes=4, in_channels=1, train_cfg=cfg)

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


def test_ga_anchor_head_loss():
    """Tests anchor head loss when truth is empty and non-empty."""
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
            ga_assigner=dict(
                type='ApproxMaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            ga_sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            center_ratio=0.2,
            ignore_ratio=0.5,
            pos_weight=-1,
            debug=False))
    head = GuidedAnchorHead(num_classes=4, in_channels=4, train_cfg=cfg)

    # Anchor head expects a multiple levels of features per image
    if torch.cuda.is_available():
        head.cuda()
        feat = [
            torch.rand(1, 4, s // (2**(i + 2)), s // (2**(i + 2))).cuda()
            for i in range(len(head.approx_anchor_generator.base_anchors))
        ]
        cls_scores, bbox_preds, shape_preds, loc_preds = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_bboxes = [torch.empty((0, 4)).cuda()]
        gt_labels = [torch.LongTensor([]).cuda()]

        gt_bboxes_ignore = None

        empty_gt_losses = head.loss(cls_scores, bbox_preds, shape_preds,
                                    loc_preds, gt_bboxes, gt_labels, img_metas,
                                    gt_bboxes_ignore)

        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert empty_box_loss.item() == 0, (
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_bboxes = [
            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
        ]
        gt_labels = [torch.LongTensor([2]).cuda()]
        one_gt_losses = head.loss(cls_scores, bbox_preds, shape_preds,
                                  loc_preds, gt_bboxes, gt_labels, img_metas,
                                  gt_bboxes_ignore)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert onegt_box_loss.item() > 0, 'box loss should be non-zero'


def test_bbox_head_loss():
    """Tests bbox head loss when truth is empty and non-empty."""
    self = BBoxHead(in_channels=8, roi_feat_size=3)

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    target_cfg = mmcv.Config(dict(pos_weight=1))

    # Test bbox loss when truth is empty
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    rois = bbox2roi([res.bboxes for res in sampling_results])
    dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) == 0, 'empty gt loss should be zero'

    # Test bbox loss when truth is non-empty
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)
    rois = bbox2roi([res.bboxes for res in sampling_results])

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) > 0, 'box-loss should be non-zero'


def test_sabl_bbox_head_loss():
    """Tests bbox head loss when truth is empty and non-empty."""
    self = SABLHead(
        num_classes=4,
        cls_in_channels=3,
        reg_in_channels=3,
        cls_out_channels=3,
        reg_offset_out_channels=3,
        reg_cls_out_channels=3,
        roi_feat_size=7)

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    target_cfg = mmcv.Config(dict(pos_weight=1))

    # Test bbox loss when truth is empty
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    rois = bbox2roi([res.bboxes for res in sampling_results])
    dummy_feats = torch.rand(num_sampled, 3, 7, 7)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox_cls',
                      0) == 0, 'empty gt bbox-cls-loss should be zero'
    assert losses.get('loss_bbox_reg',
                      0) == 0, 'empty gt bbox-reg-loss should be zero'

    # Test bbox loss when truth is non-empty
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)
    rois = bbox2roi([res.bboxes for res in sampling_results])

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 3, 7, 7)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_bbox_cls',
                      0) > 0, 'empty gt bbox-cls-loss should be zero'
    assert losses.get('loss_bbox_reg',
                      0) > 0, 'empty gt bbox-reg-loss should be zero'


def test_sabl_retina_head_loss():
    """Tests anchor head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cfg = mmcv.Config(
        dict(
            assigner=dict(
                type='ApproxMaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0.0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    head = SABLRetinaHead(
        num_classes=4,
        in_channels=3,
        feat_channels=10,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        train_cfg=cfg)
    if torch.cuda.is_available():
        head.cuda()
        # Anchor head expects a multiple levels of features per image
        feat = [
            torch.rand(1, 3, s // (2**(i + 2)), s // (2**(i + 2))).cuda()
            for i in range(len(head.approx_anchor_generator.base_anchors))
        ]
        cls_scores, bbox_preds = head.forward(feat)

        # Test that empty ground truth encourages the network
        # to predict background
        gt_bboxes = [torch.empty((0, 4)).cuda()]
        gt_labels = [torch.LongTensor([]).cuda()]

        gt_bboxes_ignore = None
        empty_gt_losses = head.loss(cls_scores, bbox_preds, gt_bboxes,
                                    gt_labels, img_metas, gt_bboxes_ignore)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_cls_loss = sum(empty_gt_losses['loss_bbox_cls'])
        empty_box_reg_loss = sum(empty_gt_losses['loss_bbox_reg'])
        assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert empty_box_cls_loss.item() == 0, (
            'there should be no box cls loss when there are no true boxes')
        assert empty_box_reg_loss.item() == 0, (
            'there should be no box reg loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should
        # be nonzero for random inputs
        gt_bboxes = [
            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
        ]
        gt_labels = [torch.LongTensor([2]).cuda()]
        one_gt_losses = head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                  img_metas, gt_bboxes_ignore)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_cls_loss = sum(one_gt_losses['loss_bbox_cls'])
        onegt_box_reg_loss = sum(one_gt_losses['loss_bbox_reg'])
        assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
        assert onegt_box_cls_loss.item() > 0, 'box loss cls should be non-zero'
        assert onegt_box_reg_loss.item() > 0, 'box loss reg should be non-zero'


def test_refine_boxes():
    """Mirrors the doctest in
    ``mmdet.models.bbox_heads.bbox_head.BBoxHead.refine_boxes`` but checks for
    multiple values of n_roi / n_img."""
    self = BBoxHead(reg_class_agnostic=True)

    test_settings = [

        # Corner case: less rois than images
        {
            'n_roi': 2,
            'n_img': 4,
            'rng': 34285940
        },

        # Corner case: no images
        {
            'n_roi': 0,
            'n_img': 0,
            'rng': 52925222
        },

        # Corner cases: few images / rois
        {
            'n_roi': 1,
            'n_img': 1,
            'rng': 1200281
        },
        {
            'n_roi': 2,
            'n_img': 1,
            'rng': 1200282
        },
        {
            'n_roi': 2,
            'n_img': 2,
            'rng': 1200283
        },
        {
            'n_roi': 1,
            'n_img': 2,
            'rng': 1200284
        },

        # Corner case: no rois few images
        {
            'n_roi': 0,
            'n_img': 1,
            'rng': 23955860
        },
        {
            'n_roi': 0,
            'n_img': 2,
            'rng': 25830516
        },

        # Corner case: no rois many images
        {
            'n_roi': 0,
            'n_img': 10,
            'rng': 671346
        },
        {
            'n_roi': 0,
            'n_img': 20,
            'rng': 699807
        },

        # Corner case: cal_similarity num rois and images
        {
            'n_roi': 20,
            'n_img': 20,
            'rng': 1200238
        },
        {
            'n_roi': 10,
            'n_img': 20,
            'rng': 1200238
        },
        {
            'n_roi': 5,
            'n_img': 5,
            'rng': 1200238
        },

        # ----------------------------------
        # Common case: more rois than images
        {
            'n_roi': 100,
            'n_img': 1,
            'rng': 337156
        },
        {
            'n_roi': 150,
            'n_img': 2,
            'rng': 275898
        },
        {
            'n_roi': 500,
            'n_img': 5,
            'rng': 4903221
        },
    ]

    for demokw in test_settings:
        try:
            n_roi = demokw['n_roi']
            n_img = demokw['n_img']
            rng = demokw['rng']

            print(f'Test refine_boxes case: {demokw!r}')
            tup = _demodata_refine_boxes(n_roi, n_img, rng=rng)
            rois, labels, bbox_preds, pos_is_gts, img_metas = tup
            bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
                                             pos_is_gts, img_metas)
            assert len(bboxes_list) == n_img
            assert sum(map(len, bboxes_list)) <= n_roi
            assert all(b.shape[1] == 4 for b in bboxes_list)
        except Exception:
            print(f'Test failed with demokw={demokw!r}')
            raise


def _demodata_refine_boxes(n_roi, n_img, rng=0):
    """Create random test data for the
    ``mmdet.models.bbox_heads.bbox_head.BBoxHead.refine_boxes`` method."""
    import numpy as np
    from mmdet.core.bbox.demodata import random_boxes
    from mmdet.core.bbox.demodata import ensure_rng
    try:
        import kwarray
    except ImportError:
        import pytest
        pytest.skip('kwarray is required for this test')
    scale = 512
    rng = ensure_rng(rng)
    img_metas = [{'img_shape': (scale, scale)} for _ in range(n_img)]
    # Create rois in the expected format
    roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
    if n_img == 0:
        assert n_roi == 0, 'cannot have any rois if there are no images'
        img_ids = torch.empty((0, ), dtype=torch.long)
        roi_boxes = torch.empty((0, 4), dtype=torch.float32)
    else:
        img_ids = rng.randint(0, n_img, (n_roi, ))
        img_ids = torch.from_numpy(img_ids)
    rois = torch.cat([img_ids[:, None].float(), roi_boxes], dim=1)
    # Create other args
    labels = rng.randint(0, 2, (n_roi, ))
    labels = torch.from_numpy(labels).long()
    bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
    # For each image, pretend random positive boxes are gts
    is_label_pos = (labels.numpy() > 0).astype(np.int)
    lbl_per_img = kwarray.group_items(is_label_pos, img_ids.numpy())
    pos_per_img = [sum(lbl_per_img.get(gid, [])) for gid in range(n_img)]
    # randomly generate with numpy then sort with torch
    _pos_is_gts = [
        rng.randint(0, 2, (npos, )).astype(np.uint8) for npos in pos_per_img
    ]
    pos_is_gts = [
        torch.from_numpy(p).sort(descending=True)[0] for p in _pos_is_gts
    ]
    return rois, labels, bbox_preds, pos_is_gts, img_metas


def test_mask_head_loss():
    """Test mask head loss when mask target is empty."""
    self = FCNMaskHead(
        num_convs=1,
        roi_feat_size=6,
        in_channels=8,
        conv_out_channels=8,
        num_classes=8)

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    # create dummy mask
    import numpy as np
    from mmdet.core import BitmapMasks
    dummy_mask = np.random.randint(0, 2, (1, 160, 240), dtype=np.uint8)
    gt_masks = [BitmapMasks(dummy_mask, 160, 240)]

    # create dummy train_cfg
    train_cfg = mmcv.Config(dict(mask_size=12, mask_thr_binary=0.5))

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8, 6, 6)

    mask_pred = self.forward(dummy_feats)
    mask_targets = self.get_targets(sampling_results, gt_masks, train_cfg)
    pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
    loss_mask = self.loss(mask_pred, mask_targets, pos_labels)

    onegt_mask_loss = sum(loss_mask['loss_mask'])
    assert onegt_mask_loss.item() > 0, 'mask loss should be non-zero'

    # test mask_iou_head
    mask_iou_head = MaskIoUHead(
        num_convs=1,
        num_fcs=1,
        roi_feat_size=6,
        in_channels=8,
        conv_out_channels=8,
        fc_out_channels=8,
        num_classes=8)

    pos_mask_pred = mask_pred[range(mask_pred.size(0)), pos_labels]
    mask_iou_pred = mask_iou_head(dummy_feats, pos_mask_pred)
    pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)), pos_labels]

    mask_iou_targets = mask_iou_head.get_targets(sampling_results, gt_masks,
                                                 pos_mask_pred, mask_targets,
                                                 train_cfg)
    loss_mask_iou = mask_iou_head.loss(pos_mask_iou_pred, mask_iou_targets)
    onegt_mask_iou_loss = loss_mask_iou['loss_mask_iou'].sum()
    assert onegt_mask_iou_loss.item() >= 0


def _dummy_bbox_sampling(proposal_list, gt_bboxes, gt_labels):
    """Create sample results that can be passed to BBoxHead.get_targets."""
    num_imgs = 1
    feat = torch.rand(1, 1, 3, 3)
    assign_config = dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        ignore_iof_thr=-1)
    sampler_config = dict(
        type='RandomSampler',
        num=512,
        pos_fraction=0.25,
        neg_pos_ub=-1,
        add_gt_as_proposals=True)
    bbox_assigner = build_assigner(assign_config)
    bbox_sampler = build_sampler(sampler_config)
    gt_bboxes_ignore = [None for _ in range(num_imgs)]
    sampling_results = []
    for i in range(num_imgs):
        assign_result = bbox_assigner.assign(proposal_list[i], gt_bboxes[i],
                                             gt_bboxes_ignore[i], gt_labels[i])
        sampling_result = bbox_sampler.sample(
            assign_result,
            proposal_list[i],
            gt_bboxes[i],
            gt_labels[i],
            feats=feat)
        sampling_results.append(sampling_result)

    return sampling_results


def test_corner_head_loss():
    """Tests corner head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    self = CornerHead(num_classes=4, in_channels=1)

    # Corner head expects a multiple levels of features per image
    feat = [
        torch.rand(1, 1, s // 4, s // 4) for _ in range(self.num_feat_levels)
    ]
    tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(tl_heats, br_heats, tl_embs, br_embs, tl_offs,
                                br_offs, gt_bboxes, gt_labels, img_metas,
                                gt_bboxes_ignore)
    empty_det_loss = sum(empty_gt_losses['det_loss'])
    empty_push_loss = sum(empty_gt_losses['push_loss'])
    empty_pull_loss = sum(empty_gt_losses['pull_loss'])
    empty_off_loss = sum(empty_gt_losses['off_loss'])
    assert empty_det_loss.item() > 0, 'det loss should be non-zero'
    assert empty_push_loss.item() == 0, (
        'there should be no push loss when there are no true boxes')
    assert empty_pull_loss.item() == 0, (
        'there should be no pull loss when there are no true boxes')
    assert empty_off_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(tl_heats, br_heats, tl_embs, br_embs, tl_offs,
                              br_offs, gt_bboxes, gt_labels, img_metas,
                              gt_bboxes_ignore)
    onegt_det_loss = sum(one_gt_losses['det_loss'])
    onegt_push_loss = sum(one_gt_losses['push_loss'])
    onegt_pull_loss = sum(one_gt_losses['pull_loss'])
    onegt_off_loss = sum(one_gt_losses['off_loss'])
    assert onegt_det_loss.item() > 0, 'det loss should be non-zero'
    assert onegt_push_loss.item() == 0, (
        'there should be no push loss when there are only one true box')
    assert onegt_pull_loss.item() > 0, 'pull loss should be non-zero'
    assert onegt_off_loss.item() > 0, 'off loss should be non-zero'

    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874],
                      [123.6667, 123.8757, 138.6326, 251.8874]]),
    ]
    gt_labels = [torch.LongTensor([2, 3])]

    # equalize the corners' embedding value of different objects to make the
    # push_loss larger than 0
    gt_bboxes_ind = (gt_bboxes[0] // 4).int().tolist()
    for tl_emb_feat, br_emb_feat in zip(tl_embs, br_embs):
        tl_emb_feat[:, :, gt_bboxes_ind[0][1],
                    gt_bboxes_ind[0][0]] = tl_emb_feat[:, :,
                                                       gt_bboxes_ind[1][1],
                                                       gt_bboxes_ind[1][0]]
        br_emb_feat[:, :, gt_bboxes_ind[0][3],
                    gt_bboxes_ind[0][2]] = br_emb_feat[:, :,
                                                       gt_bboxes_ind[1][3],
                                                       gt_bboxes_ind[1][2]]

    two_gt_losses = self.loss(tl_heats, br_heats, tl_embs, br_embs, tl_offs,
                              br_offs, gt_bboxes, gt_labels, img_metas,
                              gt_bboxes_ignore)
    twogt_det_loss = sum(two_gt_losses['det_loss'])
    twogt_push_loss = sum(two_gt_losses['push_loss'])
    twogt_pull_loss = sum(two_gt_losses['pull_loss'])
    twogt_off_loss = sum(two_gt_losses['off_loss'])
    assert twogt_det_loss.item() > 0, 'det loss should be non-zero'
    assert twogt_push_loss.item() > 0, 'push loss should be non-zero'
    assert twogt_pull_loss.item() > 0, 'pull loss should be non-zero'
    assert twogt_off_loss.item() > 0, 'off loss should be non-zero'


def test_corner_head_encode_and_decode_heatmap():
    """Tests corner head generating and decoding the heatmap."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'border': (0, 0, 0, 0)
    }]

    gt_bboxes = [
        torch.Tensor([[10, 20, 200, 240], [40, 50, 100, 200],
                      [10, 20, 200, 240]])
    ]
    gt_labels = [torch.LongTensor([1, 1, 2])]

    self = CornerHead(num_classes=4, in_channels=1, corner_emb_channels=1)

    feat = [
        torch.rand(1, 1, s // 4, s // 4) for _ in range(self.num_feat_levels)
    ]

    targets = self.get_targets(
        gt_bboxes,
        gt_labels,
        feat[0].shape,
        img_metas[0]['pad_shape'],
        with_corner_emb=self.with_corner_emb)

    gt_tl_heatmap = targets['topleft_heatmap']
    gt_br_heatmap = targets['bottomright_heatmap']
    gt_tl_offset = targets['topleft_offset']
    gt_br_offset = targets['bottomright_offset']
    embedding = targets['corner_embedding']
    [top, left], [bottom, right] = embedding[0][0]
    gt_tl_embedding_heatmap = torch.zeros([1, 1, s // 4, s // 4])
    gt_br_embedding_heatmap = torch.zeros([1, 1, s // 4, s // 4])
    gt_tl_embedding_heatmap[0, 0, top, left] = 1
    gt_br_embedding_heatmap[0, 0, bottom, right] = 1

    batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(
        tl_heat=gt_tl_heatmap,
        br_heat=gt_br_heatmap,
        tl_off=gt_tl_offset,
        br_off=gt_br_offset,
        tl_emb=gt_tl_embedding_heatmap,
        br_emb=gt_br_embedding_heatmap,
        img_meta=img_metas[0],
        k=100,
        kernel=3,
        distance_threshold=0.5)

    bboxes = batch_bboxes.view(-1, 4)
    scores = batch_scores.view(-1, 1)
    clses = batch_clses.view(-1, 1)

    idx = scores.argsort(dim=0, descending=True)
    bboxes = bboxes[idx].view(-1, 4)
    scores = scores[idx].view(-1)
    clses = clses[idx].view(-1)

    valid_bboxes = bboxes[torch.where(scores > 0.05)]
    valid_labels = clses[torch.where(scores > 0.05)]
    max_coordinate = valid_bboxes.max()
    offsets = valid_labels.to(valid_bboxes) * (max_coordinate + 1)
    gt_offsets = gt_labels[0].to(gt_bboxes[0]) * (max_coordinate + 1)

    offset_bboxes = valid_bboxes + offsets[:, None]
    offset_gtbboxes = gt_bboxes[0] + gt_offsets[:, None]

    iou_matrix = bbox_overlaps(offset_bboxes.numpy(), offset_gtbboxes.numpy())
    assert (iou_matrix == 1).sum() == 3


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
