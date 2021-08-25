import numpy as np

from mmdet.core.evaluation.mean_ap import eval_map, tpfp_default, tpfp_imagenet

det_bboxes = np.array([
    [0, 0, 10, 10],
    [10, 10, 20, 20],
    [32, 32, 38, 42],
])
gt_bboxes = np.array([[0, 0, 10, 20], [0, 10, 10, 19], [10, 10, 20, 20]])
gt_ignore = np.array([[5, 5, 10, 20], [6, 10, 10, 19]])


def test_tpfp_imagenet():

    tpfp_imagenet(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=True)
    tpfp_imagenet(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=False)


def test_tpfp_default():

    tpfp_default(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=True)
    tpfp_default(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=False)


def test_eval_map():

    # 2 image and 2 classes
    det_results = [[det_bboxes, det_bboxes], [det_bboxes, det_bboxes]]

    labels = np.array([0, 1, 1])
    labels_ignore = np.array([0, 1])
    gt_info = {
        'bboxes': gt_bboxes,
        'bboxes_ignore': gt_ignore,
        'labels': labels,
        'labels_ignore': labels_ignore
    }
    annotations = [gt_info, gt_info]
    eval_map(det_results, annotations, use_legacy_coordinate=True)
    eval_map(det_results, annotations, use_legacy_coordinate=False)
