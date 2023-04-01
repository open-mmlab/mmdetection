# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet.core.mask import BitmapMasks, PolygonMasks


def _check_fields(results, pipeline_results, keys):
    """Check data in fields from two results are same."""
    for key in keys:
        if isinstance(results[key], (BitmapMasks, PolygonMasks)):
            assert np.equal(results[key].to_ndarray(),
                            pipeline_results[key].to_ndarray()).all()
        else:
            assert np.equal(results[key], pipeline_results[key]).all()
            assert results[key].dtype == pipeline_results[key].dtype


def check_result_same(results, pipeline_results):
    """Check whether the `pipeline_results` is the same with the predefined
    `results`.

    Args:
        results (dict): Predefined results which should be the standard output
            of the transform pipeline.
        pipeline_results (dict): Results processed by the transform pipeline.
    """
    # check image
    _check_fields(results, pipeline_results,
                  results.get('img_fields', ['img']))
    # check bboxes
    _check_fields(results, pipeline_results, results.get('bbox_fields', []))
    # check masks
    _check_fields(results, pipeline_results, results.get('mask_fields', []))
    # check segmentations
    _check_fields(results, pipeline_results, results.get('seg_fields', []))
    # check gt_labels
    if 'gt_labels' in results:
        assert np.equal(results['gt_labels'],
                        pipeline_results['gt_labels']).all()


def construct_toy_data(poly2mask=True):
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    # image
    results['img'] = img
    results['img_shape'] = img.shape
    results['img_fields'] = ['img']
    # bboxes
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    results['gt_bboxes'] = np.array([[0., 0., 2., 1.]], dtype=np.float32)
    results['gt_bboxes_ignore'] = np.array([[2., 0., 3., 1.]],
                                           dtype=np.float32)
    # labels
    results['gt_labels'] = np.array([1], dtype=np.int64)
    # masks
    results['mask_fields'] = ['gt_masks']
    if poly2mask:
        gt_masks = np.array([[0, 1, 1, 0], [0, 1, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results['gt_masks'] = BitmapMasks(gt_masks, 2, 4)
    else:
        raw_masks = [[np.array([0, 0, 2, 0, 2, 1, 0, 1], dtype=np.float)]]
        results['gt_masks'] = PolygonMasks(raw_masks, 2, 4)
    # segmentations
    results['seg_fields'] = ['gt_semantic_seg']
    results['gt_semantic_seg'] = img[..., 0]
    return results


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
