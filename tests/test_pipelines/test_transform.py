import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch
from mmcv.utils import build_from_cfg

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.builder import PIPELINES


def test_resize():
    # test assertion if img_scale is a list
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', img_scale=[1333, 800], keep_ratio=True)
        build_from_cfg(transform, PIPELINES)

    # test assertion if len(img_scale) while ratio_range is not None
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            img_scale=[(1333, 800), (1333, 600)],
            ratio_range=(0.9, 1.1),
            keep_ratio=True)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid multiscale_mode
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            img_scale=[(1333, 800), (1333, 600)],
            keep_ratio=True,
            multiscale_mode='2333')
        build_from_cfg(transform, PIPELINES)

    # test assertion if both scale and scale_factor are setted
    with pytest.raises(AssertionError):
        results = dict(
            img_prefix=osp.join(osp.dirname(__file__), '../data'),
            img_info=dict(filename='color.jpg'))
        load = dict(type='LoadImageFromFile')
        load = build_from_cfg(load, PIPELINES)
        transform = dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
        transform = build_from_cfg(transform, PIPELINES)
        results = load(results)
        results['scale'] = (1333, 800)
        results['scale_factor'] = 1.0
        results = transform(results)

    transform = dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()

    results.pop('scale')
    results.pop('scale_factor')
    transform = dict(
        type='Resize',
        img_scale=(1280, 800),
        multiscale_mode='value',
        keep_ratio=False)
    resize_module = build_from_cfg(transform, PIPELINES)
    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (800, 1280, 3)


def test_flip():
    # test assertion for invalid flip_ratio
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', flip_ratio=1.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomFlip', flip_ratio=1, direction='horizonta')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='RandomFlip', flip_ratio=1)
    flip_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img', 'img2']

    results = flip_module(results)
    assert np.equal(results['img'], results['img2']).all()

    flip_module = build_from_cfg(transform, PIPELINES)
    results = flip_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert np.equal(original_img, results['img']).all()


def test_random_crop():
    # test assertion for invalid random crop
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCrop', crop_size=(-1, 0))
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    results['img'] = img

    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # TODO: add img_fields test
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    def create_random_bboxes(num_bboxes, img_w, img_h):
        bboxes_left_top = np.random.uniform(0, 0.5, size=(num_bboxes, 2))
        bboxes_right_bottom = np.random.uniform(0.5, 1, size=(num_bboxes, 2))
        bboxes = np.concatenate((bboxes_left_top, bboxes_right_bottom), 1)
        bboxes = (bboxes * np.array([img_w, img_h, img_w, img_h])).astype(
            np.int)
        return bboxes

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='RandomCrop', crop_size=(h - 20, w - 20))
    crop_module = build_from_cfg(transform, PIPELINES)
    results = crop_module(results)
    assert results['img'].shape[:2] == (h - 20, w - 20)
    # All bboxes should be reserved after crop
    assert results['img_shape'][:2] == (h - 20, w - 20)
    assert results['gt_bboxes'].shape[0] == 8
    assert results['gt_bboxes_ignore'].shape[0] == 2

    def area(bboxes):
        return np.prod(bboxes[:, 2:4] - bboxes[:, 0:2], axis=1)

    assert (area(results['gt_bboxes']) <= area(gt_bboxes)).all()
    assert (area(results['gt_bboxes_ignore']) <= area(gt_bboxes_ignore)).all()


def test_min_iou_random_crop():

    def create_random_bboxes(num_bboxes, img_w, img_h):
        bboxes_left_top = np.random.uniform(0, 0.5, size=(num_bboxes, 2))
        bboxes_right_bottom = np.random.uniform(0.5, 1, size=(num_bboxes, 2))
        bboxes = np.concatenate((bboxes_left_top, bboxes_right_bottom), 1)
        bboxes = (bboxes * np.array([img_w, img_h, img_w, img_h])).astype(
            np.int)
        return bboxes

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    results['img'] = img

    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(1, w, h)
    gt_bboxes_ignore = create_random_bboxes(1, w, h)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='MinIoURandomCrop')
    crop_module = build_from_cfg(transform, PIPELINES)

    # Test for img_fields
    results_test = copy.deepcopy(results)
    results_test['img1'] = results_test['img']
    results_test['img_fields'] = ['img', 'img1']
    with pytest.raises(AssertionError):
        crop_module(results_test)
    results = crop_module(results)
    patch = np.array([0, 0, results['img_shape'][1], results['img_shape'][0]])
    ious = bbox_overlaps(patch.reshape(-1, 4),
                         results['gt_bboxes']).reshape(-1)
    ious_ignore = bbox_overlaps(
        patch.reshape(-1, 4), results['gt_bboxes_ignore']).reshape(-1)
    mode = crop_module.mode
    if mode == 1:
        assert np.equal(results['gt_bboxes'], gt_bboxes).all()
        assert np.equal(results['gt_bboxes_ignore'], gt_bboxes_ignore).all()
    else:
        assert (ious >= mode).all()
        assert (ious_ignore >= mode).all()


def test_pad():
    # test assertion if both size_divisor and size is None
    with pytest.raises(AssertionError):
        transform = dict(type='Pad')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='Pad', size_divisor=32)
    transform = build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img', 'img2']

    results = transform(results)
    assert np.equal(results['img'], results['img2']).all()
    # original img already divisible by 32
    assert np.equal(results['img'], original_img).all()
    img_shape = results['img'].shape
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0

    resize_transform = dict(
        type='Resize', img_scale=(1333, 800), keep_ratio=True)
    resize_module = build_from_cfg(resize_transform, PIPELINES)
    results = resize_module(results)
    results = transform(results)
    img_shape = results['img'].shape
    assert np.equal(results['img'], results['img2']).all()
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0


def test_normalize():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    transform = dict(type='Normalize', **img_norm_cfg)
    transform = build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img', 'img2']

    results = transform(results)
    assert np.equal(results['img'], results['img2']).all()

    mean = np.array(img_norm_cfg['mean'])
    std = np.array(img_norm_cfg['std'])
    converted_img = (original_img[..., ::-1] - mean) / std
    assert np.allclose(results['img'], converted_img)


def test_albu_transform():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../data'),
        img_info=dict(filename='color.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)

    albu_transform = dict(
        type='Albu', transforms=[dict(type='ChannelShuffle', p=1)])
    albu_transform = build_from_cfg(albu_transform, PIPELINES)

    normalize = dict(type='Normalize', mean=[0] * 3, std=[0] * 3, to_rgb=True)
    normalize = build_from_cfg(normalize, PIPELINES)

    # Execute transforms
    results = load(results)
    results = albu_transform(results)
    results = normalize(results)

    assert results['img'].dtype == np.float32


def test_random_center_crop_pad():
    # test assertion for invalid crop_size while test_mode=False
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(-1, 0),
            test_mode=False,
            test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid ratios while test_mode=False
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(511, 511),
            ratios=(1.0),
            test_mode=False,
            test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid mean, std and to_rgb
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(511, 511),
            mean=None,
            std=None,
            to_rgb=None,
            test_mode=False,
            test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid crop_size while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(511, 511),
            ratios=None,
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid ratios while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=None,
            ratios=(0.9, 1.0, 1.1),
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid border while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=None,
            ratios=None,
            border=128,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid test_pad_mode while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=None,
            ratios=None,
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('do_nothing', 100))
        build_from_cfg(transform, PIPELINES)

    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../data'),
        img_info=dict(filename='color.jpg'))

    load = dict(type='LoadImageFromFile', to_float32=True)
    load = build_from_cfg(load, PIPELINES)
    results = load(results)
    test_results = copy.deepcopy(results)

    def create_random_bboxes(num_bboxes, img_w, img_h):
        bboxes_left_top = np.random.uniform(0, 0.5, size=(num_bboxes, 2))
        bboxes_right_bottom = np.random.uniform(0.5, 1, size=(num_bboxes, 2))
        bboxes = np.concatenate((bboxes_left_top, bboxes_right_bottom), 1)
        bboxes = (bboxes * np.array([img_w, img_h, img_w, img_h])).astype(
            np.int)
        return bboxes

    h, w, _ = results['img_shape']
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    train_transform = dict(
        type='RandomCenterCropPad',
        crop_size=(h - 20, w - 20),
        ratios=(1.0, ),
        border=128,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        test_mode=False,
        test_pad_mode=None)
    crop_module = build_from_cfg(train_transform, PIPELINES)
    train_results = crop_module(results)
    assert train_results['img'].shape[:2] == (h - 20, w - 20)
    # All bboxes should be reserved after crop
    assert train_results['pad_shape'][:2] == (h - 20, w - 20)
    assert train_results['gt_bboxes'].shape[0] == 8
    assert train_results['gt_bboxes_ignore'].shape[0] == 2

    test_transform = dict(
        type='RandomCenterCropPad',
        crop_size=None,
        ratios=None,
        border=None,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        test_mode=True,
        test_pad_mode=('logical_or', 127))
    crop_module = build_from_cfg(test_transform, PIPELINES)

    test_results = crop_module(test_results)
    assert test_results['img'].shape[:2] == (h | 127, w | 127)
    assert test_results['pad_shape'][:2] == (h | 127, w | 127)
    assert 'border' in test_results


def test_multi_scale_flip_aug():
    # test assertion if give both scale_factor and img_scale
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            scale_factor=1.0,
            img_scale=[(1333, 800)],
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    # test assertion if both scale_factor and img_scale are None
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            scale_factor=None,
            img_scale=None,
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    # test assertion if img_scale is not tuple or list of tuple
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            img_scale=[1333, 800],
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    # test assertion if flip_direction is not str or list of str
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            img_scale=[(1333, 800)],
            flip_direction=1,
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    scale_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800), (1333, 640)],
        transforms=[dict(type='Resize', keep_ratio=True)])
    transform = build_from_cfg(scale_transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img']

    scale_results = transform(copy.deepcopy(results))
    assert len(scale_results['img']) == 2
    assert scale_results['img'][0].shape == (750, 1333, 3)
    assert scale_results['img_shape'][0] == (750, 1333, 3)
    assert scale_results['img'][1].shape == (640, 1138, 3)
    assert scale_results['img_shape'][1] == (640, 1138, 3)

    scale_factor_transform = dict(
        type='MultiScaleFlipAug',
        scale_factor=[0.8, 1.0, 1.2],
        transforms=[dict(type='Resize', keep_ratio=False)])
    transform = build_from_cfg(scale_factor_transform, PIPELINES)
    scale_factor_results = transform(copy.deepcopy(results))
    assert len(scale_factor_results['img']) == 3
    assert scale_factor_results['img'][0].shape == (230, 409, 3)
    assert scale_factor_results['img_shape'][0] == (230, 409, 3)
    assert scale_factor_results['img'][1].shape == (288, 512, 3)
    assert scale_factor_results['img_shape'][1] == (288, 512, 3)
    assert scale_factor_results['img'][2].shape == (345, 614, 3)
    assert scale_factor_results['img_shape'][2] == (345, 614, 3)

    # test pipeline of coco_detection
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../data'),
        img_info=dict(filename='color.jpg'))
    load_cfg, multi_scale_cfg = mmcv.Config.fromfile(
        'configs/_base_/datasets/coco_detection.py').test_pipeline
    load = build_from_cfg(load_cfg, PIPELINES)
    transform = build_from_cfg(multi_scale_cfg, PIPELINES)
    results = transform(load(results))
    assert len(results['img']) == 1
    assert len(results['img_metas']) == 1
    assert isinstance(results['img'][0], torch.Tensor)
    assert isinstance(results['img_metas'][0], mmcv.parallel.DataContainer)
    assert results['img_metas'][0].data['ori_shape'] == (288, 512, 3)
    assert results['img_metas'][0].data['img_shape'] == (750, 1333, 3)
    assert results['img_metas'][0].data['pad_shape'] == (768, 1344, 3)
    assert results['img_metas'][0].data['scale_factor'].tolist() == [
        2.603515625, 2.6041667461395264, 2.603515625, 2.6041667461395264
    ]
