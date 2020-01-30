"""
Tests the Assigner objects.

CommandLine:
    pytest tests/test_assigner.py
    xdoctest tests/test_assigner.py zero



"""
import torch

from mmdet.core import MaxIoUAssigner
from mmdet.core.bbox.assigners import ApproxMaxIoUAssigner, PointAssigner


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
        [32, 32, 38, 42],
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
    """
    Test corner case where an image might have no true detections
    """
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
    gt_bboxes = torch.FloatTensor([])
    assign_result = self.assign(bboxes, gt_bboxes)

    expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
    assert torch.all(assign_result.gt_inds == expected_gt_inds)


def test_max_iou_assigner_with_empty_boxes():
    """
    Test corner case where an network might predict no boxes
    """
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


def test_max_iou_assigner_with_empty_boxes_and_gt():
    """
    Test corner case where an network might predict no boxes and no gt
    """
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
    """
    Test corner case where an image might have no true detections
    """
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
    """
    Test corner case where an image might predict no points and no gt
    """
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
    """
    Test corner case where an image might have no true detections
    """
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
    """
    Test corner case where an network might predict no boxes
    """
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
    """
    Test corner case where an network might predict no boxes and no gt
    """
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
    """
    Test random instantiation of assign result to catch corner cases
    """
    from mmdet.core.bbox.assigners.assign_result import AssignResult
    AssignResult.random()

    AssignResult.random(num_gts=0, num_preds=0)
    AssignResult.random(num_gts=0, num_preds=3)
    AssignResult.random(num_gts=3, num_preds=3)
    AssignResult.random(num_gts=0, num_preds=3)
    AssignResult.random(num_gts=7, num_preds=7)
    AssignResult.random(num_gts=7, num_preds=64)
    AssignResult.random(num_gts=24, num_preds=3)
