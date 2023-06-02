# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np
from mmcv.utils import build_from_cfg
from numpy.testing import assert_array_equal

from mmdet.datasets.builder import PIPELINES
from .utils import construct_toy_data


def test_adjust_color():
    results = construct_toy_data()
    # test wighout aug
    transform = dict(type='ColorTransform', prob=0, level=10)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])

    # test with factor 1
    img = results['img']
    transform = dict(type='ColorTransform', prob=1, level=10)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], img)

    # test with factor 0
    transform_module.factor = 0
    img_gray = mmcv.bgr2gray(img.copy())
    img_r = np.stack([img_gray, img_gray, img_gray], axis=-1)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], img_r)

    # test with factor 0.5
    transform_module.factor = 0.5
    results_transformed = transform_module(copy.deepcopy(results))
    img = results['img']
    assert_array_equal(
        results_transformed['img'],
        np.round(np.clip((img * 0.5 + img_r * 0.5), 0, 255)).astype(img.dtype))


def test_imequalize(nb_rand_test=100):

    def _imequalize(img):
        # equalize the image using PIL.ImageOps.equalize
        from PIL import Image, ImageOps
        img = Image.fromarray(img)
        equalized_img = np.asarray(ImageOps.equalize(img))
        return equalized_img

    results = construct_toy_data()
    # test wighout aug
    transform = dict(type='EqualizeTransform', prob=0)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])

    # test equalize with case step=0
    transform = dict(type='EqualizeTransform', prob=1.)
    transform_module = build_from_cfg(transform, PIPELINES)
    img = np.array([[0, 0, 0], [120, 120, 120], [255, 255, 255]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results['img'] = img
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], img)

    # test equalize with randomly sampled image.
    for _ in range(nb_rand_test):
        img = np.clip(np.random.uniform(0, 1, (1000, 1200, 3)) * 260, 0,
                      255).astype(np.uint8)
        results['img'] = img
        results_transformed = transform_module(copy.deepcopy(results))
        assert_array_equal(results_transformed['img'], _imequalize(img))


def test_adjust_brightness(nb_rand_test=100):

    def _adjust_brightness(img, factor):
        # adjust the brightness of image using
        # PIL.ImageEnhance.Brightness
        from PIL import Image
        from PIL.ImageEnhance import Brightness
        img = Image.fromarray(img)
        brightened_img = Brightness(img).enhance(factor)
        return np.asarray(brightened_img)

    results = construct_toy_data()
    # test wighout aug
    transform = dict(type='BrightnessTransform', level=10, prob=0)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])

    # test case with factor 1.0
    transform = dict(type='BrightnessTransform', level=10, prob=1.)
    transform_module = build_from_cfg(transform, PIPELINES)
    transform_module.factor = 1.0
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])

    # test case with factor 0.0
    transform_module.factor = 0.0
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'],
                       np.zeros_like(results['img']))

    # test with randomly sampled images and factors.
    for _ in range(nb_rand_test):
        img = np.clip(np.random.uniform(0, 1, (1000, 1200, 3)) * 260, 0,
                      255).astype(np.uint8)
        factor = np.random.uniform()
        transform_module.factor = factor
        results['img'] = img
        np.testing.assert_allclose(
            transform_module(copy.deepcopy(results))['img'].astype(np.int32),
            _adjust_brightness(img, factor).astype(np.int32),
            rtol=0,
            atol=1)


def test_adjust_contrast(nb_rand_test=100):

    def _adjust_contrast(img, factor):
        from PIL import Image
        from PIL.ImageEnhance import Contrast

        # Image.fromarray defaultly supports RGB, not BGR.
        # convert from BGR to RGB
        img = Image.fromarray(img[..., ::-1], mode='RGB')
        contrasted_img = Contrast(img).enhance(factor)
        # convert from RGB to BGR
        return np.asarray(contrasted_img)[..., ::-1]

    results = construct_toy_data()
    # test wighout aug
    transform = dict(type='ContrastTransform', level=10, prob=0)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])

    # test case with factor 1.0
    transform = dict(type='ContrastTransform', level=10, prob=1.)
    transform_module = build_from_cfg(transform, PIPELINES)
    transform_module.factor = 1.0
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])

    # test case with factor 0.0
    transform_module.factor = 0.0
    results_transformed = transform_module(copy.deepcopy(results))
    np.testing.assert_allclose(
        results_transformed['img'],
        _adjust_contrast(results['img'], 0.),
        rtol=0,
        atol=1)

    # test adjust_contrast with randomly sampled images and factors.
    for _ in range(nb_rand_test):
        img = np.clip(np.random.uniform(0, 1, (1200, 1000, 3)) * 260, 0,
                      255).astype(np.uint8)
        factor = np.random.uniform()
        transform_module.factor = factor
        results['img'] = img
        results_transformed = transform_module(copy.deepcopy(results))
        # Note the gap (less_equal 1) between PIL.ImageEnhance.Contrast
        # and mmcv.adjust_contrast comes from the gap that converts from
        # a color image to gray image using mmcv or PIL.
        np.testing.assert_allclose(
            transform_module(copy.deepcopy(results))['img'].astype(np.int32),
            _adjust_contrast(results['img'], factor).astype(np.int32),
            rtol=0,
            atol=1)
