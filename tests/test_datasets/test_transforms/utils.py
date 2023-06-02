# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.testing import assert_allclose

from mmdet.structures.bbox import BaseBoxes, HorizontalBoxes
from mmdet.structures.mask import BitmapMasks, PolygonMasks


def create_random_bboxes(num_bboxes, img_w, img_h):
    bboxes_left_top = np.random.uniform(0, 0.5, size=(num_bboxes, 2))
    bboxes_right_bottom = np.random.uniform(0.5, 1, size=(num_bboxes, 2))
    bboxes = np.concatenate((bboxes_left_top, bboxes_right_bottom), 1)
    bboxes = (bboxes * np.array([img_w, img_h, img_w, img_h])).astype(
        np.float32)
    return bboxes


def create_full_masks(gt_bboxes, img_w, img_h):
    xmin, ymin = gt_bboxes[:, 0:1], gt_bboxes[:, 1:2]
    xmax, ymax = gt_bboxes[:, 2:3], gt_bboxes[:, 3:4]
    gt_masks = np.zeros((len(gt_bboxes), img_h, img_w), dtype=np.uint8)
    for i in range(len(gt_bboxes)):
        gt_masks[i, int(ymin[i]):int(ymax[i]), int(xmin[i]):int(xmax[i])] = 1
    gt_masks = BitmapMasks(gt_masks, img_h, img_w)
    return gt_masks


def construct_toy_data(poly2mask, use_box_type=False):
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    results['img'] = img
    results['img_shape'] = img.shape[:2]
    if use_box_type:
        results['gt_bboxes'] = HorizontalBoxes(
            np.array([[1, 0, 2, 2]], dtype=np.float32))
    else:
        results['gt_bboxes'] = np.array([[1, 0, 2, 2]], dtype=np.float32)
    results['gt_bboxes_labels'] = np.array([13], dtype=np.int64)
    if poly2mask:
        gt_masks = np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
    else:
        raw_masks = [[np.array([1, 2, 1, 0, 2, 1], dtype=np.float32)]]
        results['gt_masks'] = PolygonMasks(raw_masks, 3, 4)
    results['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
    results['gt_seg_map'] = np.array(
        [[255, 13, 255, 255], [255, 13, 13, 255], [255, 13, 255, 255]],
        dtype=np.uint8)
    return results


def check_result_same(results, pipeline_results, check_keys):
    """Check whether the ``pipeline_results`` is the same with the predefined
    ``results``.

    Args:
        results (dict): Predefined results which should be the standard
            output of the transform pipeline.
        pipeline_results (dict): Results processed by the transform
            pipeline.
        check_keys (tuple): Keys that need to be checked between
            results and pipeline_results.
    """
    for key in check_keys:
        if results.get(key, None) is None:
            continue
        if isinstance(results[key], (BitmapMasks, PolygonMasks)):
            assert_allclose(pipeline_results[key].to_ndarray(),
                            results[key].to_ndarray())
        elif isinstance(results[key], BaseBoxes):
            assert_allclose(pipeline_results[key].tensor, results[key].tensor)
        else:
            assert_allclose(pipeline_results[key], results[key])
