import numpy as np

from mmdet.core.evaluation.recall import eval_recalls

det_bboxes = np.array([
    [0, 0, 10, 10],
    [10, 10, 20, 20],
    [32, 32, 38, 42],
])
gt_bboxes = np.array([[0, 0, 10, 20], [0, 10, 10, 19], [10, 10, 20, 20]])
gt_ignore = np.array([[5, 5, 10, 20], [6, 10, 10, 19]])


def test_eval_recalls():
    gts = [gt_bboxes, gt_bboxes, gt_bboxes]
    proposals = [det_bboxes, det_bboxes, det_bboxes]

    recall = eval_recalls(
        gts, proposals, proposal_nums=2, use_legacy_coordinate=True)
    assert recall.shape == (1, 1)
    assert 0.66 < recall[0][0] < 0.667
    recall = eval_recalls(
        gts, proposals, proposal_nums=2, use_legacy_coordinate=False)
    assert recall.shape == (1, 1)
    assert 0.66 < recall[0][0] < 0.667

    recall = eval_recalls(
        gts, proposals, proposal_nums=2, use_legacy_coordinate=True)
    assert recall.shape == (1, 1)
    assert 0.66 < recall[0][0] < 0.667
    recall = eval_recalls(
        gts,
        proposals,
        iou_thrs=[0.1, 0.9],
        proposal_nums=2,
        use_legacy_coordinate=False)
    assert recall.shape == (1, 2)
    assert recall[0][1] <= recall[0][0]
    recall = eval_recalls(
        gts,
        proposals,
        iou_thrs=[0.1, 0.9],
        proposal_nums=2,
        use_legacy_coordinate=True)
    assert recall.shape == (1, 2)
    assert recall[0][1] <= recall[0][0]
