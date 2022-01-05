import numpy as np

from mmdet.core.evaluation.mean_ap import (eval_map, tpfp_default,
                                           tpfp_imagenet, tpfp_openimages)

det_bboxes = np.array([
    [0, 0, 10, 10],
    [10, 10, 20, 20],
    [32, 32, 38, 42],
])
gt_bboxes = np.array([[0, 0, 10, 20], [0, 10, 10, 19], [10, 10, 20, 20]])
gt_ignore = np.array([[5, 5, 10, 20], [6, 10, 10, 19]])


def test_tpfp_imagenet():

    result = tpfp_imagenet(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=True)
    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()

    result = tpfp_imagenet(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=False)
    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()


def test_tpfp_default():

    result = tpfp_default(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=True)

    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()
    result = tpfp_default(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        use_legacy_coordinate=False)

    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()


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
    mean_ap, eval_results = eval_map(
        det_results, annotations, use_legacy_coordinate=True)
    assert 0.291 < mean_ap < 0.293
    eval_map(det_results, annotations, use_legacy_coordinate=False)
    assert 0.291 < mean_ap < 0.293


def test_tpfp_openimages():

    det_bboxes = np.array([[10, 10, 15, 15, 1.0], [15, 15, 30, 30, 0.98],
                           [10, 10, 25, 25, 0.98], [28, 28, 35, 35, 0.97],
                           [30, 30, 51, 51, 0.96], [100, 110, 120, 130, 0.15]])
    gt_bboxes = np.array([[10., 10., 30., 30.], [30., 30., 50., 50.]])
    gt_groups_of = np.array([True, False], dtype=np.bool)
    gt_ignore = np.zeros((0, 4))

    # Open Images evaluation using group of.
    result = tpfp_openimages(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        gt_bboxes_group_of=gt_groups_of,
        use_group_of=True,
        ioa_thr=0.5)

    tp = result[0]
    fp = result[1]
    cls_dets = result[2]

    assert tp.shape == (1, 4)
    assert fp.shape == (1, 4)
    assert cls_dets.shape == (4, 5)

    assert (tp == np.array([[0, 1, 0, 1]])).all()
    assert (fp == np.array([[1, 0, 1, 0]])).all()
    cls_dets_gt = np.array([[28., 28., 35., 35., 0.97],
                            [30., 30., 51., 51., 0.96],
                            [100., 110., 120., 130., 0.15],
                            [10., 10., 15., 15., 1.]])
    assert (cls_dets == cls_dets_gt).all()

    # Open Images evaluation not using group of.
    result = tpfp_openimages(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        gt_bboxes_group_of=gt_groups_of,
        use_group_of=False,
        ioa_thr=0.5)
    tp = result[0]
    fp = result[1]
    cls_dets = result[2]
    assert tp.shape == (1, 6)
    assert fp.shape == (1, 6)
    assert cls_dets.shape == (6, 5)

    # Open Images evaluation using group of, and gt is all group of bboxes.
    gt_groups_of = np.array([True, True], dtype=np.bool)
    result = tpfp_openimages(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        gt_bboxes_group_of=gt_groups_of,
        use_group_of=True,
        ioa_thr=0.5)
    tp = result[0]
    fp = result[1]
    cls_dets = result[2]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert cls_dets.shape == (3, 5)

    # Open Images evaluation with empty gt.
    gt_bboxes = np.zeros((0, 4))
    gt_groups_of = np.empty((0))
    result = tpfp_openimages(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_ignore,
        gt_bboxes_group_of=gt_groups_of,
        use_group_of=True,
        ioa_thr=0.5)
    fp = result[1]
    assert (fp == np.array([[1, 1, 1, 1, 1, 1]])).all()
