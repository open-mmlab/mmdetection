import copy
import os.path as osp

import numpy as np
import pytest
from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import PIPELINES


def test_translate():
    # test assertion for invalid value of level
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=-1)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid type of level
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=[1])
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid prob
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, prob=-0.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for the num of elements in tuple fill_val
    with pytest.raises(AssertionError):
        transform = dict(
            type='Translate', level=1, fill_val=(128, 128, 128, 128))
        build_from_cfg(transform, PIPELINES)

    # test ValueError for invalid type of fill_val
    with pytest.raises(ValueError):
        transform = dict(type='Translate', level=1, fill_val=[128, 128, 128])
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of fill_val
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, fill_val=(128, -1, 256))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of axis
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, fill_val=128, axis='z')
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid type of max_translate_offset
    with pytest.raises(AssertionError):
        transform = dict(
            type='Translate',
            level=1,
            fill_val=128,
            max_translate_offset=(250., ))
        build_from_cfg(transform, PIPELINES)

    # test case when level=0 (without translate aug)
    transform = dict(
        type='Translate', level=0, prob=1.0, fill_val=(104, 116, 124))
    translate_module = build_from_cfg(transform, PIPELINES)
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../data'),
        img_info=dict(filename='color.jpg'))
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)
    results = load(results)
    results_translated = translate_module(copy.deepcopy(results))
    assert np.equal(results['img'], results_translated['img']).all()

    # test case when level>0 and translate along x-axis (left shift).
    fill_val = (104, 116, 124)
    transform = dict(
        type='Translate', level=8, prob=1.0, fill_val=fill_val, axis='x')
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    results_translated = translate_module(
        copy.deepcopy(results), neg_offset_prob=1.0)
    h, w, c = results['img'].shape
    pad_arr = np.stack([
        np.ones((h, offset)) * fill_val[0],
        np.ones((h, offset)) * fill_val[1],
        np.ones((h, offset)) * fill_val[2]
    ],
                       axis=-1).astype(results['img'].dtype)  # (h, w-offset,3)
    gt_translated = np.concatenate((results['img'][:, offset:, :], pad_arr),
                                   axis=1)
    assert np.equal(results_translated['img'], gt_translated).all()
    assert results_translated['img'].dtype == results['img'].dtype

    # test case when level>0 and translate along x-axis (right shift).
    results_translated = translate_module(
        copy.deepcopy(results), neg_offset_prob=0.0)
    gt_translated = np.concatenate((pad_arr, results['img'][:, :-offset, :]),
                                   axis=1)
    assert np.equal(results_translated['img'], gt_translated).all()

    # test case when level>0 and translate along y-axis (top shift).
    transform = dict(
        type='Translate', level=10, prob=1.0, fill_val=fill_val, axis='y')
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    results_translated = translate_module(
        copy.deepcopy(results), neg_offset_prob=1.0)
    pad_arr = np.stack([
        np.ones((offset, w)) * fill_val[0],
        np.ones((offset, w)) * fill_val[1],
        np.ones((offset, w)) * fill_val[2]
    ],
                       axis=-1).astype(results['img'].dtype)
    gt_translated = np.concatenate((results['img'][offset:, :, :], pad_arr),
                                   axis=0)
    assert np.equal(results_translated['img'], gt_translated).all()
    assert results_translated['img'].dtype == results['img'].dtype

    # test case when level>0 and translate along y-axis (bottom shift).
    results_translated = translate_module(
        copy.deepcopy(results), neg_offset_prob=0.0)
    gt_translated = np.concatenate((pad_arr, results['img'][:-offset, :, :]),
                                   axis=0)
    assert np.equal(results_translated['img'], gt_translated).all()
