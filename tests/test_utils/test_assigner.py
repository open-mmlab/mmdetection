# Copyright (c) OpenMMLab. All rights reserved.
"""Tests the Assigner objects.

CommandLine:
    pytest tests/test_utils/test_assigner.py
    xdoctest tests/test_utils/test_assigner.py zero
"""
import pytest
import torch

from mmdet.core.bbox.assigners import (ApproxMaxIoUAssigner,
                                       AscendMaxIoUAssigner,
                                       CenterRegionAssigner, HungarianAssigner,
                                       MaskHungarianAssigner, MaxIoUAssigner,
                                       PointAssigner, SimOTAAssigner,
                                       TaskAlignedAssigner, UniformAssigner)


def test_max_iou_assigner():
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([2, 3])
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 4
    assert len(assign_result.labels) == 4

    expected_gt_inds = torch.LongTensor([1, 0, 2, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_max_iou_assigner_with_ignore():
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [30, 32, 40, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_bboxes_ignore = torch.Tensor([
        [30, 30, 40, 40],
    ])
    assign_result = self.assign(
        bboxes, gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore)

    expected_gt_inds = torch.LongTensor([1, 0, 2, -1])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_max_iou_assigner_with_empty_gt():
    """Test corner case where an image might have no true detections."""
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.empty(0, 4)
    assign_result = self.assign(bboxes, gt_bboxes)

    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_max_iou_assigner_with_empty_boxes():
    """Test corner case where a network might predict no boxes."""
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.empty((0, 4))
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([2, 3])

    # Test with gt_labels
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 0
    assert tuple(assign_result.labels.shape) == (0, )

    # Test without gt_labels
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=None)
    assert len(assign_result.gt_inds) == 0
    assert assign_result.labels is None


def test_max_iou_assigner_with_empty_boxes_and_ignore():
    """Test corner case where a network might predict no boxes and
    ignore_iof_thr is on."""
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
    )
    bboxes = torch.empty((0, 4))
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_bboxes_ignore = torch.Tensor([
        [30, 30, 40, 40],
    ])
    gt_labels = torch.LongTensor([2, 3])

    # Test with gt_labels
    assign_result = self.assign(
        bboxes,
        gt_bboxes,
        gt_labels=gt_labels,
        gt_bboxes_ignore=gt_bboxes_ignore)
    assert len(assign_result.gt_inds) == 0
    assert tuple(assign_result.labels.shape) == (0, )

    # Test without gt_labels
    assign_result = self.assign(
        bboxes, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore)
    assert len(assign_result.gt_inds) == 0
    assert assign_result.labels is None


def test_max_iou_assigner_with_empty_boxes_and_gt():
    """Test corner case where a network might predict no boxes and no gt."""
    self = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.empty((0, 4))
    gt_bboxes = torch.empty((0, 4))
    assign_result = self.assign(bboxes, gt_bboxes)
    assert len(assign_result.gt_inds) == 0


def test_point_assigner():
    self = PointAssigner()
    points = torch.FloatTensor([  # [x, y, stride]
        [0, 0, 1],
        [10, 10, 1],
        [5, 5, 1],
        [32, 32, 1],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    assign_result = self.assign(points, gt_bboxes)
    expected_gt_inds = torch.LongTensor([1, 2, 1, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_point_assigner_with_empty_gt():
    """Test corner case where an image might have no true detections."""
    self = PointAssigner()
    points = torch.FloatTensor([  # [x, y, stride]
        [0, 0, 1],
        [10, 10, 1],
        [5, 5, 1],
        [32, 32, 1],
    ])
    gt_bboxes = torch.FloatTensor([])
    assign_result = self.assign(points, gt_bboxes)

    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_point_assigner_with_empty_boxes_and_gt():
    """Test corner case where an image might predict no points and no gt."""
    self = PointAssigner()
    points = torch.FloatTensor([])
    gt_bboxes = torch.FloatTensor([])
    assign_result = self.assign(points, gt_bboxes)
    assert len(assign_result.gt_inds) == 0


def test_approx_iou_assigner():
    self = ApproxMaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    approxs_per_octave = 1
    approxs = bboxes
    squares = bboxes
    assign_result = self.assign(approxs, squares, approxs_per_octave,
                                gt_bboxes)

    expected_gt_inds = torch.LongTensor([1, 0, 2, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_approx_iou_assigner_with_empty_gt():
    """Test corner case where an image might have no true detections."""
    self = ApproxMaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([])
    approxs_per_octave = 1
    approxs = bboxes
    squares = bboxes
    assign_result = self.assign(approxs, squares, approxs_per_octave,
                                gt_bboxes)

    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_approx_iou_assigner_with_empty_boxes():
    """Test corner case where an network might predict no boxes."""
    self = ApproxMaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.empty((0, 4))
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    approxs_per_octave = 1
    approxs = bboxes
    squares = bboxes
    assign_result = self.assign(approxs, squares, approxs_per_octave,
                                gt_bboxes)
    assert len(assign_result.gt_inds) == 0


def test_approx_iou_assigner_with_empty_boxes_and_gt():
    """Test corner case where an network might predict no boxes and no gt."""
    self = ApproxMaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    bboxes = torch.empty((0, 4))
    gt_bboxes = torch.empty((0, 4))
    approxs_per_octave = 1
    approxs = bboxes
    squares = bboxes
    assign_result = self.assign(approxs, squares, approxs_per_octave,
                                gt_bboxes)
    assert len(assign_result.gt_inds) == 0


def test_random_assign_result():
    """Test random instantiation of assign result to catch corner cases."""
    from mmdet.core.bbox.assigners.assign_result import AssignResult
    AssignResult.random()

    AssignResult.random(num_gts=0, num_preds=0)
    AssignResult.random(num_gts=0, num_preds=3)
    AssignResult.random(num_gts=3, num_preds=3)
    AssignResult.random(num_gts=0, num_preds=3)
    AssignResult.random(num_gts=7, num_preds=7)
    AssignResult.random(num_gts=7, num_preds=64)
    AssignResult.random(num_gts=24, num_preds=3)


def test_center_region_assigner():
    self = CenterRegionAssigner(pos_scale=0.3, neg_scale=1)
    bboxes = torch.FloatTensor([[0, 0, 10, 10], [10, 10, 20, 20], [8, 8, 9,
                                                                   9]])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 11, 11],  # match bboxes[0]
        [10, 10, 20, 20],  # match bboxes[1]
        [4.5, 4.5, 5.5, 5.5],  # match bboxes[0] but area is too small
        [0, 0, 10, 10],  # match bboxes[1] and has a smaller area than gt[0]
    ])
    gt_labels = torch.LongTensor([2, 3, 4, 5])
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 3
    assert len(assign_result.labels) == 3
    expected_gt_inds = torch.LongTensor([4, 2, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)
    shadowed_labels = assign_result.get_extra_property('shadowed_labels')
    # [8, 8, 9, 9] in the shadowed region of [0, 0, 11, 11] (label: 2)
    assert torch.any(shadowed_labels == torch.LongTensor([[2, 2]]))
    # [8, 8, 9, 9] in the shadowed region of [0, 0, 10, 10] (label: 5)
    assert torch.any(shadowed_labels == torch.LongTensor([[2, 5]]))
    # [0, 0, 10, 10] is already assigned to [4.5, 4.5, 5.5, 5.5].
    #   Therefore, [0, 0, 11, 11] (label: 2) is shadowed
    assert torch.any(shadowed_labels == torch.LongTensor([[0, 2]]))


def test_center_region_assigner_with_ignore():
    self = CenterRegionAssigner(
        pos_scale=0.5,
        neg_scale=1,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 10],  # match bboxes[0]
        [10, 10, 20, 20],  # match bboxes[1]
    ])
    gt_bboxes_ignore = torch.FloatTensor([
        [0, 0, 10, 10],  # match bboxes[0]
    ])
    gt_labels = torch.LongTensor([1, 2])
    assign_result = self.assign(
        bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_bboxes_ignore,
        gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 2
    assert len(assign_result.labels) == 2

    expected_gt_inds = torch.LongTensor([-1, 2])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_center_region_assigner_with_empty_bboxes():
    self = CenterRegionAssigner(
        pos_scale=0.5,
        neg_scale=1,
    )
    bboxes = torch.empty((0, 4)).float()
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 10],  # match bboxes[0]
        [10, 10, 20, 20],  # match bboxes[1]
    ])
    gt_labels = torch.LongTensor([1, 2])
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
    assert assign_result.gt_inds is None or assign_result.gt_inds.numel() == 0
    assert assign_result.labels is None or assign_result.labels.numel() == 0


def test_center_region_assigner_with_empty_gts():
    self = CenterRegionAssigner(
        pos_scale=0.5,
        neg_scale=1,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
    ])
    gt_bboxes = torch.empty((0, 4)).float()
    gt_labels = torch.empty((0, )).long()
    assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 2
    expected_gt_inds = torch.LongTensor([0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_hungarian_match_assigner():
    self = HungarianAssigner()
    assert self.iou_cost.iou_mode == 'giou'

    # test no gt bboxes
    bbox_pred = torch.rand((10, 4))
    cls_pred = torch.rand((10, 81))
    gt_bboxes = torch.empty((0, 4)).float()
    gt_labels = torch.empty((0, )).long()
    img_meta = dict(img_shape=(10, 8, 3))
    assign_result = self.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels,
                                img_meta)
    assert torch.all(assign_result.gt_inds == 0)
    assert torch.all(assign_result.labels == -1)

    # test with gt bboxes
    gt_bboxes = torch.FloatTensor([[0, 0, 5, 7], [3, 5, 7, 8]])
    gt_labels = torch.LongTensor([1, 20])
    assign_result = self.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels,
                                img_meta)

    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_bboxes.size(0)
    assert (assign_result.labels > -1).sum() == gt_bboxes.size(0)

    # test iou mode
    self = HungarianAssigner(
        iou_cost=dict(type='IoUCost', iou_mode='iou', weight=1.0))
    assert self.iou_cost.iou_mode == 'iou'
    assign_result = self.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_bboxes.size(0)
    assert (assign_result.labels > -1).sum() == gt_bboxes.size(0)

    # test focal loss mode
    self = HungarianAssigner(
        iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
        cls_cost=dict(type='FocalLossCost', weight=1.))
    assert self.iou_cost.iou_mode == 'giou'
    assign_result = self.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_bboxes.size(0)
    assert (assign_result.labels > -1).sum() == gt_bboxes.size(0)


def test_uniform_assigner():
    self = UniformAssigner(0.15, 0.7, 1)
    pred_bbox = torch.FloatTensor([
        [1, 1, 12, 8],
        [4, 4, 20, 20],
        [1, 5, 15, 15],
        [30, 5, 32, 42],
    ])
    anchor = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([2, 3])
    assign_result = self.assign(
        pred_bbox, anchor, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 4
    assert len(assign_result.labels) == 4

    expected_gt_inds = torch.LongTensor([-1, 0, 2, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_uniform_assigner_with_empty_gt():
    """Test corner case where an image might have no true detections."""
    self = UniformAssigner(0.15, 0.7, 1)
    pred_bbox = torch.FloatTensor([
        [1, 1, 12, 8],
        [4, 4, 20, 20],
        [1, 5, 15, 15],
        [30, 5, 32, 42],
    ])
    anchor = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.empty(0, 4)
    assign_result = self.assign(pred_bbox, anchor, gt_bboxes)

    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_uniform_assigner_with_empty_boxes():
    """Test corner case where a network might predict no boxes."""
    self = UniformAssigner(0.15, 0.7, 1)
    pred_bbox = torch.empty((0, 4))
    anchor = torch.empty((0, 4))
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([2, 3])

    # Test with gt_labels
    assign_result = self.assign(
        pred_bbox, anchor, gt_bboxes, gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 0
    assert tuple(assign_result.labels.shape) == (0, )

    # Test without gt_labels
    assign_result = self.assign(pred_bbox, anchor, gt_bboxes, gt_labels=None)
    assert len(assign_result.gt_inds) == 0


def test_sim_ota_assigner():
    self = SimOTAAssigner(
        center_radius=2.5, candidate_topk=1, iou_weight=3.0, cls_weight=1.0)
    pred_scores = torch.FloatTensor([[0.2], [0.8]])
    priors = torch.Tensor([[0, 12, 23, 34], [4, 5, 6, 7]])
    decoded_bboxes = torch.Tensor([[[30, 40, 50, 60]], [[4, 5, 6, 7]]])
    gt_bboxes = torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])
    gt_labels = torch.LongTensor([2])
    assign_result = self.assign(pred_scores, priors, decoded_bboxes, gt_bboxes,
                                gt_labels)

    expected_gt_inds = torch.LongTensor([0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_task_aligned_assigner():
    with pytest.raises(AssertionError):
        TaskAlignedAssigner(topk=0)

    self = TaskAlignedAssigner(topk=13)
    pred_score = torch.FloatTensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4],
                                    [0.4, 0.5]])
    pred_bbox = torch.FloatTensor([
        [1, 1, 12, 8],
        [4, 4, 20, 20],
        [1, 5, 15, 15],
        [30, 5, 32, 42],
    ])
    anchor = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([0, 1])
    assign_result = self.assign(
        pred_score,
        pred_bbox,
        anchor,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels)
    assert len(assign_result.gt_inds) == 4
    assert len(assign_result.labels) == 4

    # test empty gt
    gt_bboxes = torch.empty(0, 4)
    gt_labels = torch.empty(0, 2)
    assign_result = self.assign(
        pred_score, pred_bbox, anchor, gt_bboxes=gt_bboxes)
    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_mask_hungarian_match_assigner():
    # test no gt masks
    assigner_cfg = dict(
        cls_cost=dict(type='ClassificationCost', weight=1.0),
        mask_cost=dict(type='FocalLossCost', weight=20.0, binary_input=True),
        dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0))
    self = MaskHungarianAssigner(**assigner_cfg)
    cls_pred = torch.rand((10, 133))
    mask_pred = torch.rand((10, 50, 50))

    gt_labels = torch.empty((0, )).long()
    gt_masks = torch.empty((0, 50, 50)).float()
    img_meta = None
    assign_result = self.assign(cls_pred, mask_pred, gt_labels, gt_masks,
                                img_meta)
    assert torch.all(assign_result.gt_inds == 0)
    assert torch.all(assign_result.labels == -1)

    # test with gt masks of naive_dice is True
    gt_labels = torch.LongTensor([10, 100])
    gt_masks = torch.zeros((2, 50, 50)).long()
    gt_masks[0, :25] = 1
    gt_masks[0, 25:] = 1
    assign_result = self.assign(cls_pred, mask_pred, gt_labels, gt_masks,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_labels.size(0)
    assert (assign_result.labels > -1).sum() == gt_labels.size(0)

    # test with cls mode
    assigner_cfg = dict(
        cls_cost=dict(type='ClassificationCost', weight=1.0),
        mask_cost=dict(type='FocalLossCost', weight=0.0, binary_input=True),
        dice_cost=dict(type='DiceCost', weight=0.0, pred_act=True, eps=1.0))
    self = MaskHungarianAssigner(**assigner_cfg)
    assign_result = self.assign(cls_pred, mask_pred, gt_labels, gt_masks,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_labels.size(0)
    assert (assign_result.labels > -1).sum() == gt_labels.size(0)

    # test with mask focal mode
    assigner_cfg = dict(
        cls_cost=dict(type='ClassificationCost', weight=0.0),
        mask_cost=dict(type='FocalLossCost', weight=1.0, binary_input=True),
        dice_cost=dict(type='DiceCost', weight=0.0, pred_act=True, eps=1.0))
    self = MaskHungarianAssigner(**assigner_cfg)
    assign_result = self.assign(cls_pred, mask_pred, gt_labels, gt_masks,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_labels.size(0)
    assert (assign_result.labels > -1).sum() == gt_labels.size(0)

    # test with mask dice mode
    assigner_cfg = dict(
        cls_cost=dict(type='ClassificationCost', weight=0.0),
        mask_cost=dict(type='FocalLossCost', weight=0.0, binary_input=True),
        dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0))
    self = MaskHungarianAssigner(**assigner_cfg)
    assign_result = self.assign(cls_pred, mask_pred, gt_labels, gt_masks,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_labels.size(0)
    assert (assign_result.labels > -1).sum() == gt_labels.size(0)

    # test with mask dice mode that naive_dice is False
    assigner_cfg = dict(
        cls_cost=dict(type='ClassificationCost', weight=0.0),
        mask_cost=dict(type='FocalLossCost', weight=0.0, binary_input=True),
        dice_cost=dict(
            type='DiceCost',
            weight=1.0,
            pred_act=True,
            eps=1.0,
            naive_dice=False))
    self = MaskHungarianAssigner(**assigner_cfg)
    assign_result = self.assign(cls_pred, mask_pred, gt_labels, gt_masks,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_labels.size(0)
    assert (assign_result.labels > -1).sum() == gt_labels.size(0)

    # test with mask bce mode
    assigner_cfg = dict(
        cls_cost=dict(type='ClassificationCost', weight=0.0),
        mask_cost=dict(
            type='CrossEntropyLossCost', weight=1.0, use_sigmoid=True),
        dice_cost=dict(type='DiceCost', weight=0.0, pred_act=True, eps=1.0))
    self = MaskHungarianAssigner(**assigner_cfg)
    assign_result = self.assign(cls_pred, mask_pred, gt_labels, gt_masks,
                                img_meta)
    assert torch.all(assign_result.gt_inds > -1)
    assert (assign_result.gt_inds > 0).sum() == gt_labels.size(0)
    assert (assign_result.labels > -1).sum() == gt_labels.size(0)

    # test with ce mode of CrossEntropyLossCost which is not supported yet
    assigner_cfg = dict(
        cls_cost=dict(type='ClassificationCost', weight=0.0),
        mask_cost=dict(
            type='CrossEntropyLossCost', weight=1.0, use_sigmoid=False),
        dice_cost=dict(type='DiceCost', weight=0.0, pred_act=True, eps=1.0))
    with pytest.raises(AssertionError):
        self = MaskHungarianAssigner(**assigner_cfg)


def test_ascend_max_iou_assigner():
    self = AscendMaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
    batch_bboxes = torch.FloatTensor([[
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ]])
    batch_gt_bboxes = torch.FloatTensor([[
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ]])
    batch_gt_labels = torch.LongTensor([[2, 3]])
    batch_bboxes_ignore_mask = torch.IntTensor([[1, 1, 1, 1]])
    assign_result = self.assign(
        batch_bboxes,
        batch_gt_bboxes,
        batch_gt_labels=batch_gt_labels,
        batch_bboxes_ignore_mask=batch_bboxes_ignore_mask)

    expected_batch_pos_mask = torch.IntTensor([1, 0, 1, 0])
    expected_batch_anchor_gt_indes = torch.IntTensor([0, 0, 1, 0])
    expected_batch_anchor_gt_labels = torch.IntTensor([2, 0, 3, 0])

    assert torch.all(assign_result.batch_pos_mask == expected_batch_pos_mask)
    assert torch.all(
        assign_result.batch_anchor_gt_indes *
        assign_result.batch_pos_mask == expected_batch_anchor_gt_indes)
    assert torch.all(
        assign_result.batch_anchor_gt_labels *
        assign_result.batch_pos_mask == expected_batch_anchor_gt_labels)
