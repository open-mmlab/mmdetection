import copy
import os.path as osp

import numpy as np
import pytest
from mmcv.utils import build_from_cfg

from mmdet.core.mask import BitmapMasks
from mmdet.datasets.builder import DATASETS, PIPELINES


def construct_cocodata_example():
    # construct CocoDataset as the dataset example for testing
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='LoadAnnotations',
            with_bbox=True,
            with_mask=True,
            with_seg=True,
            poly2mask=True),
    ]
    data_root = osp.join(osp.dirname(__file__), '../data/coco_dummy/')
    data = dict(
        type='CocoDataset',
        ann_file=osp.join(data_root, 'annotations/instances_val2017.json'),
        img_prefix=osp.join(data_root, 'val2017/'),
        seg_prefix=osp.join(data_root, 'stuffthingmaps/val2017/'),
        pipeline=pipeline,
        test_mode=False)
    return build_from_cfg(data, DATASETS)


def _check_keys(results, results_shared):
    assert len(set(results.keys()).difference(set(results_shared.keys()))) == 0
    assert len(set(results_shared.keys()).difference(set(results.keys()))) == 0


def _check_fields(results, results_shared, keys):
    for key in keys:
        if isinstance(results[key], BitmapMasks):
            assert np.equal(results[key].to_ndarray(),
                            results_shared[key].to_ndarray()).all()
        else:
            assert np.equal(results[key], results_shared[key]).all()


def check_shear(results, results_shared):
    _check_keys(results, results_shared)
    # check image
    _check_fields(results, results_shared, results.get('img_fields', ['img']))
    # check bboxes
    _check_fields(results, results_shared, results.get('bbox_fields', []))
    # check masks
    _check_fields(results, results_shared, results.get('mask_fields', []))
    # check segmentations
    _check_fields(results, results_shared, results.get('seg_fields', []))


def test_shear():
    # test assertion for invalid type of max_shear_magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', level=1, max_shear_magnitude=(0.5, ))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of max_shear_magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', level=2, max_shear_magnitude=1.2)
        build_from_cfg(transform, PIPELINES)

    coco_dataset = construct_cocodata_example()
    # randomly sample one image and load the results
    results = coco_dataset[np.random.choice(len(coco_dataset))]

    # test case when no shear aug (level=0)
    img_fill_val = (104, 116, 124)
    seg_ignore_label = 255
    transform = dict(
        type='Shear',
        level=0,
        prob=1.,
        img_fill_val=img_fill_val,
        seg_ignore_label=seg_ignore_label,
        axis='x')
    shear_module = build_from_cfg(transform, PIPELINES)
    results_wo_shear = shear_module(copy.deepcopy(results))
    check_shear(results, results_wo_shear)

    # test case when no shear aug (prob<=0)
    transform = dict(
        type='Shear', level=10, prob=0., img_fill_val=img_fill_val, axis='y')
    shear_module = build_from_cfg(transform, PIPELINES)
    results_wo_shear = shear_module(copy.deepcopy(results))
    check_shear(results, results_wo_shear)


test_shear()
