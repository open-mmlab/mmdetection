import copy

import numpy as np
import pytest
from mmcv.utils import build_from_cfg

from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES


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


def _check_fields(results, results_rotated, keys):
    for key in keys:
        if isinstance(results[key], (BitmapMasks, PolygonMasks)):
            assert np.equal(results[key].to_ndarray(),
                            results_rotated[key].to_ndarray()).all()
        else:
            assert np.equal(results[key], results_rotated[key]).all()


def check_rotate(results, results_rotated):
    # check image
    _check_fields(results, results_rotated, results.get('img_fields', ['img']))
    # check bboxes
    _check_fields(results, results_rotated, results.get('bbox_fields', []))
    # check masks
    _check_fields(results, results_rotated, results.get('mask_fields', []))
    # check segmentations
    _check_fields(results, results_rotated, results.get('seg_fields', []))
    # _check gt_labels
    if 'gt_labels' in results:
        assert np.equal(results['gt_labels'],
                        results_rotated['gt_labels']).all()


def test_rotate():
    # test assertion for invalid type of max_rotate_angle
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', level=1, max_rotate_angle=(30, ))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid type of scale
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', level=2, scale=(1.2, ))
        build_from_cfg(transform, PIPELINES)

    # test ValueError for invalid type of img_fill_val
    with pytest.raises(ValueError):
        transform = dict(
            type='Rotate', level=2, img_fill_val=[
                128,
            ])
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid number of elements in center
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', level=2, center=(0.5, ))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid type of center
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', level=2, center=[0, 0])
        build_from_cfg(transform, PIPELINES)

    # test case when no rotate aug (level=0)
    results = construct_toy_data()
    img_fill_val = (104, 116, 124)
    seg_ignore_label = 255
    transform = dict(
        type='Rotate',
        level=0,
        prob=1.,
        img_fill_val=img_fill_val,
        seg_ignore_label=seg_ignore_label,
    )
    rotate_module = build_from_cfg(transform, PIPELINES)
    results_wo_rotate = rotate_module(copy.deepcopy(results))
    check_rotate(results, results_wo_rotate)

    # test case when no rotate aug (prob<=0)
    transform = dict(
        type='Rotate', level=10, prob=0., img_fill_val=img_fill_val, scale=0.6)
    rotate_module = build_from_cfg(transform, PIPELINES)
    results_wo_rotate = rotate_module(copy.deepcopy(results))
    check_rotate(results, results_wo_rotate)

    # test clockwise rotation with angle 90
    results = construct_toy_data()
    img_fill_val = 128
    transform = dict(
        type='Rotate',
        level=10,
        max_rotate_angle=90,
        img_fill_val=img_fill_val,
        # set random_negative_prob to 0 for clockwise rotation
        random_negative_prob=0.,
        prob=1.)
    rotate_module = build_from_cfg(transform, PIPELINES)
    results_rotated = rotate_module(copy.deepcopy(results))
    img_r = np.array([[img_fill_val, 6, 2, img_fill_val],
                      [img_fill_val, 7, 3, img_fill_val]]).astype(np.uint8)
    img_r = np.stack([img_r, img_r, img_r], axis=-1)
    results_gt = copy.deepcopy(results)
    results_gt['img'] = img_r
    results_gt['gt_bboxes'] = np.array([[1., 0., 2., 1.]], dtype=np.float32)
    results_gt['gt_bboxes_ignore'] = np.empty((0, 4), dtype=np.float32)
    gt_masks = np.array([[0, 1, 1, 0], [0, 0, 1, 0]],
                        dtype=np.uint8)[None, :, :]
    results_gt['gt_masks'] = BitmapMasks(gt_masks, 2, 4)
    results_gt['gt_semantic_seg'] = np.array(
        [[255, 6, 2, 255], [255, 7, 3,
                            255]]).astype(results['gt_semantic_seg'].dtype)
    check_rotate(results_gt, results_rotated)

    # test clockwise rotation with angle 90, PolygonMasks
    results = construct_toy_data(poly2mask=False)
    results_rotated = rotate_module(copy.deepcopy(results))
    gt_masks = [[np.array([2, 0, 2, 1, 1, 1, 1, 0], dtype=np.float)]]
    results_gt['gt_masks'] = PolygonMasks(gt_masks, 2, 4)
    check_rotate(results_gt, results_rotated)

    # test counter-clockwise roatation with angle 90,
    # and specify the ratation center
    img_fill_val = (104, 116, 124)
    transform = dict(
        type='Rotate',
        level=10,
        max_rotate_angle=90,
        center=(0, 0),
        img_fill_val=img_fill_val,
        # set random_negative_prob to 1 for counter-clockwise rotation
        random_negative_prob=1.,
        prob=1.)
    results = construct_toy_data()
    rotate_module = build_from_cfg(transform, PIPELINES)
    results_rotated = rotate_module(copy.deepcopy(results))
    results_gt = copy.deepcopy(results)
    h, w = results['img'].shape[:2]
    img_r = np.stack([
        np.ones((h, w)) * img_fill_val[0],
        np.ones((h, w)) * img_fill_val[1],
        np.ones((h, w)) * img_fill_val[2]
    ],
                     axis=-1).astype(np.uint8)
    img_r[0, 0, :] = 1
    img_r[0, 1, :] = 5
    results_gt['img'] = img_r
    results_gt['gt_bboxes'] = np.empty((0, 4), dtype=np.float32)
    results_gt['gt_bboxes_ignore'] = np.empty((0, 4), dtype=np.float32)
    results_gt['gt_labels'] = np.empty((0, ), dtype=np.int64)
    gt_masks = np.empty((0, h, w), dtype=np.uint8)
    results_gt['gt_masks'] = BitmapMasks(gt_masks, h, w)
    gt_seg = (np.ones((h, w)) * 255).astype(results['gt_semantic_seg'].dtype)
    gt_seg[0, 0], gt_seg[0, 1] = 1, 5
    results_gt['gt_semantic_seg'] = gt_seg
    check_rotate(results_gt, results_rotated)

    transform = dict(
        type='Rotate',
        level=10,
        max_rotate_angle=90,
        center=(0),
        img_fill_val=img_fill_val,
        random_negative_prob=1.,
        prob=1.)
    rotate_module = build_from_cfg(transform, PIPELINES)
    results_rotated = rotate_module(copy.deepcopy(results))
    check_rotate(results_gt, results_rotated)

    # test counter-clockwise roatation with angle 90,
    # and specify the ratation center, PolygonMasks
    results = construct_toy_data(poly2mask=False)
    results_rotated = rotate_module(copy.deepcopy(results))
    gt_masks = [[np.array([0, 0, 0, 0, 1, 0, 1, 0], dtype=np.float)]]
    results_gt['gt_masks'] = PolygonMasks(gt_masks, 2, 4)
    check_rotate(results_gt, results_rotated)

    # test AutoAugment equipped with Rotate
    policies = [[dict(type='Rotate', level=10, prob=1.)]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))

    policies = [[
        dict(type='Rotate', level=10, prob=1.),
        dict(
            type='Rotate',
            level=8,
            max_rotate_angle=90,
            center=(0),
            img_fill_val=img_fill_val)
    ]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))
