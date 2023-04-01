# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import pytest
from mmcv.utils import build_from_cfg

from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from .utils import check_result_same, construct_toy_data


def test_shear():
    # test assertion for invalid type of max_shear_magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', level=1, max_shear_magnitude=(0.5, ))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of max_shear_magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', level=2, max_shear_magnitude=1.2)
        build_from_cfg(transform, PIPELINES)

    # test ValueError for invalid type of img_fill_val
    with pytest.raises(ValueError):
        transform = dict(type='Shear', level=2, img_fill_val=[128])
        build_from_cfg(transform, PIPELINES)

    results = construct_toy_data()
    # test case when no shear aug (level=0, direction='horizontal')
    img_fill_val = (104, 116, 124)
    seg_ignore_label = 255
    transform = dict(
        type='Shear',
        level=0,
        prob=1.,
        img_fill_val=img_fill_val,
        seg_ignore_label=seg_ignore_label,
        direction='horizontal')
    shear_module = build_from_cfg(transform, PIPELINES)
    results_wo_shear = shear_module(copy.deepcopy(results))
    check_result_same(results, results_wo_shear)

    # test case when no shear aug (level=0, direction='vertical')
    transform = dict(
        type='Shear',
        level=0,
        prob=1.,
        img_fill_val=img_fill_val,
        seg_ignore_label=seg_ignore_label,
        direction='vertical')
    shear_module = build_from_cfg(transform, PIPELINES)
    results_wo_shear = shear_module(copy.deepcopy(results))
    check_result_same(results, results_wo_shear)

    # test case when no shear aug (prob<=0)
    transform = dict(
        type='Shear',
        level=10,
        prob=0.,
        img_fill_val=img_fill_val,
        direction='vertical')
    shear_module = build_from_cfg(transform, PIPELINES)
    results_wo_shear = shear_module(copy.deepcopy(results))
    check_result_same(results, results_wo_shear)

    # test shear horizontally, magnitude=1
    transform = dict(
        type='Shear',
        level=10,
        prob=1.,
        img_fill_val=img_fill_val,
        direction='horizontal',
        max_shear_magnitude=1.,
        random_negative_prob=0.)
    shear_module = build_from_cfg(transform, PIPELINES)
    results_sheared = shear_module(copy.deepcopy(results))
    results_gt = copy.deepcopy(results)
    img_s = np.array([[1, 2, 3, 4], [0, 5, 6, 7]], dtype=np.uint8)
    img_s = np.stack([img_s, img_s, img_s], axis=-1)
    img_s[1, 0, :] = np.array(img_fill_val)
    results_gt['img'] = img_s
    results_gt['gt_bboxes'] = np.array([[0., 0., 3., 1.]], dtype=np.float32)
    results_gt['gt_bboxes_ignore'] = np.array([[2., 0., 4., 1.]],
                                              dtype=np.float32)
    gt_masks = np.array([[0, 1, 1, 0], [0, 0, 1, 0]],
                        dtype=np.uint8)[None, :, :]
    results_gt['gt_masks'] = BitmapMasks(gt_masks, 2, 4)
    results_gt['gt_semantic_seg'] = np.array(
        [[1, 2, 3, 4], [255, 5, 6, 7]], dtype=results['gt_semantic_seg'].dtype)
    check_result_same(results_gt, results_sheared)

    # test PolygonMasks with shear horizontally, magnitude=1
    results = construct_toy_data(poly2mask=False)
    results_sheared = shear_module(copy.deepcopy(results))
    print(results_sheared['gt_masks'])
    gt_masks = [[np.array([0, 0, 2, 0, 3, 1, 1, 1], dtype=np.float)]]
    results_gt['gt_masks'] = PolygonMasks(gt_masks, 2, 4)
    check_result_same(results_gt, results_sheared)

    # test shear vertically, magnitude=-1
    img_fill_val = 128
    results = construct_toy_data()
    transform = dict(
        type='Shear',
        level=10,
        prob=1.,
        img_fill_val=img_fill_val,
        direction='vertical',
        max_shear_magnitude=1.,
        random_negative_prob=1.)
    shear_module = build_from_cfg(transform, PIPELINES)
    results_sheared = shear_module(copy.deepcopy(results))
    results_gt = copy.deepcopy(results)
    img_s = np.array([[1, 6, img_fill_val, img_fill_val],
                      [5, img_fill_val, img_fill_val, img_fill_val]],
                     dtype=np.uint8)
    img_s = np.stack([img_s, img_s, img_s], axis=-1)
    results_gt['img'] = img_s
    results_gt['gt_bboxes'] = np.empty((0, 4), dtype=np.float32)
    results_gt['gt_labels'] = np.empty((0, ), dtype=np.int64)
    results_gt['gt_bboxes_ignore'] = np.empty((0, 4), dtype=np.float32)
    gt_masks = np.array([[0, 1, 0, 0], [0, 0, 0, 0]],
                        dtype=np.uint8)[None, :, :]
    results_gt['gt_masks'] = BitmapMasks(gt_masks, 2, 4)
    results_gt['gt_semantic_seg'] = np.array(
        [[1, 6, 255, 255], [5, 255, 255, 255]],
        dtype=results['gt_semantic_seg'].dtype)
    check_result_same(results_gt, results_sheared)

    # test PolygonMasks with shear vertically, magnitude=-1
    results = construct_toy_data(poly2mask=False)
    results_sheared = shear_module(copy.deepcopy(results))
    gt_masks = [[np.array([0, 0, 2, 0, 2, 0, 0, 1], dtype=np.float)]]
    results_gt['gt_masks'] = PolygonMasks(gt_masks, 2, 4)
    check_result_same(results_gt, results_sheared)

    results = construct_toy_data()
    # same mask for BitmapMasks and PolygonMasks
    results['gt_masks'] = BitmapMasks(
        np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=np.uint8)[None, :, :], 2,
        4)
    results['gt_bboxes'] = np.array([[1., 0., 2., 1.]], dtype=np.float32)
    results_sheared_bitmap = shear_module(copy.deepcopy(results))
    check_result_same(results_sheared_bitmap, results_sheared)

    # test AutoAugment equipped with Shear
    policies = [[dict(type='Shear', level=10, prob=1.)]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))

    policies = [[
        dict(type='Shear', level=10, prob=1.),
        dict(
            type='Shear',
            level=8,
            img_fill_val=img_fill_val,
            direction='vertical',
            max_shear_magnitude=1.)
    ]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))
