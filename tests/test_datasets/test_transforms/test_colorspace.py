# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

from mmdet.datasets.transforms import (AutoContrast, Brightness, Color,
                                       ColorTransform, Contrast, Equalize,
                                       Invert, Posterize, Sharpness, Solarize,
                                       SolarizeAdd)
from .utils import check_result_same, construct_toy_data


class TestColorTransform(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_colortransform(self):
        # test assertion for invalid value of level
        with self.assertRaises(AssertionError):
            transform = ColorTransform(level=-1)

        # test assertion for invalid prob
        with self.assertRaises(AssertionError):
            transform = ColorTransform(level=1, prob=-0.5)

        # test case when no translation is called (prob=0)
        transform = ColorTransform(prob=0.0, level=10)
        results_wo_color = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_color, self.check_keys)

    def test_repr(self):
        transform = ColorTransform(level=10, prob=1.)
        self.assertEqual(
            repr(transform), ('ColorTransform(prob=1.0, '
                              'level=10, '
                              'min_mag=0.1, '
                              'max_mag=1.9)'))


class TestColor(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_color(self):
        # test case when level=5 (without color aug)
        transform = Color(prob=1.0, level=5)
        results_wo_color = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_color, self.check_keys)
        # test case when level=0
        transform = Color(prob=1.0, level=0)
        transform(copy.deepcopy(self.results_mask))
        # test case when level=10
        transform = Color(prob=1.0, level=10)
        transform(copy.deepcopy(self.results_mask))

    def test_repr(self):
        transform = Color(level=10, prob=1.)
        self.assertEqual(
            repr(transform), ('Color(prob=1.0, '
                              'level=10, '
                              'min_mag=0.1, '
                              'max_mag=1.9)'))


class TestBrightness(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_brightness(self):
        # test case when level=5 (without Brightness aug)
        transform = Brightness(level=5, prob=1.0)
        results_wo_brightness = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_brightness,
                          self.check_keys)
        # test case when level=0
        transform = Brightness(prob=1.0, level=0)
        transform(copy.deepcopy(self.results_mask))
        # test case when level=10
        transform = Brightness(prob=1.0, level=10)
        transform(copy.deepcopy(self.results_mask))

    def test_repr(self):
        transform = Brightness(prob=1.0, level=10)
        self.assertEqual(
            repr(transform), ('Brightness(prob=1.0, '
                              'level=10, '
                              'min_mag=0.1, '
                              'max_mag=1.9)'))


class TestContrast(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_contrast(self):
        # test case when level=5 (without Contrast aug)
        transform = Contrast(prob=1.0, level=5)
        results_wo_contrast = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_contrast,
                          self.check_keys)
        # test case when level=0
        transform = Contrast(prob=1.0, level=0)
        transform(copy.deepcopy(self.results_mask))
        # test case when level=10
        transform = Contrast(prob=1.0, level=10)
        transform(copy.deepcopy(self.results_mask))

    def test_repr(self):
        transform = Contrast(level=10, prob=1.)
        self.assertEqual(
            repr(transform), ('Contrast(prob=1.0, '
                              'level=10, '
                              'min_mag=0.1, '
                              'max_mag=1.9)'))


class TestSharpness(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_sharpness(self):
        # test case when level=5 (without Sharpness aug)
        transform = Sharpness(prob=1.0, level=5)
        results_wo_sharpness = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_sharpness,
                          self.check_keys)
        # test case when level=0
        transform = Sharpness(prob=1.0, level=0)
        transform(copy.deepcopy(self.results_mask))
        # test case when level=10
        transform = Sharpness(prob=1.0, level=10)
        transform(copy.deepcopy(self.results_mask))

    def test_repr(self):
        transform = Sharpness(level=10, prob=1.)
        self.assertEqual(
            repr(transform), ('Sharpness(prob=1.0, '
                              'level=10, '
                              'min_mag=0.1, '
                              'max_mag=1.9)'))


class TestSolarize(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_solarize(self):
        # test case when level=10 (without Solarize aug)
        transform = Solarize(prob=1.0, level=10)
        results_wo_solarize = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_solarize,
                          self.check_keys)
        # test case when level=0
        transform = Solarize(prob=1.0, level=0)
        transform(copy.deepcopy(self.results_mask))

    def test_repr(self):
        transform = Solarize(level=10, prob=1.)
        self.assertEqual(
            repr(transform), ('Solarize(prob=1.0, '
                              'level=10, '
                              'min_mag=0.0, '
                              'max_mag=256.0)'))


class TestSolarizeAdd(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_solarize(self):
        # test case when level=0 (without Solarize aug)
        transform = SolarizeAdd(prob=1.0, level=0)
        results_wo_solarizeadd = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_solarizeadd,
                          self.check_keys)
        # test case when level=10
        transform = SolarizeAdd(prob=1.0, level=10)
        transform(copy.deepcopy(self.results_mask))

    def test_repr(self):
        transform = SolarizeAdd(level=10, prob=1.)
        self.assertEqual(
            repr(transform), ('SolarizeAdd(prob=1.0, '
                              'level=10, '
                              'min_mag=0.0, '
                              'max_mag=110.0)'))


class TestPosterize(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_posterize(self):
        # test case when level=10 (without Posterize aug)
        transform = Posterize(prob=1.0, level=10, max_mag=8.0)
        results_wo_posterize = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_posterize,
                          self.check_keys)
        # test case when level=0
        transform = Posterize(prob=1.0, level=0)
        transform(copy.deepcopy(self.results_mask))

    def test_repr(self):
        transform = Posterize(level=10, prob=1.)
        self.assertEqual(
            repr(transform), ('Posterize(prob=1.0, '
                              'level=10, '
                              'min_mag=0.0, '
                              'max_mag=4.0)'))


class TestEqualize(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_equalize(self):
        # test case when no translation is called (prob=0)
        transform = Equalize(prob=0.0)
        results_wo_equalize = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_equalize,
                          self.check_keys)
        # test case when translation is called
        transform = Equalize(prob=1.0)
        transform(copy.deepcopy(self.results_mask))

    def test_repr(self):
        transform = Equalize(prob=1.0)
        self.assertEqual(
            repr(transform), ('Equalize(prob=1.0, '
                              'level=None, '
                              'min_mag=0.1, '
                              'max_mag=1.9)'))


class TestAutoContrast(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_autocontrast(self):
        # test case when no translation is called (prob=0)
        transform = AutoContrast(prob=0.0)
        results_wo_autocontrast = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_autocontrast,
                          self.check_keys)
        # test case when translation is called
        transform = AutoContrast(prob=1.0)
        transform(copy.deepcopy(self.results_mask))

    def test_repr(self):
        transform = AutoContrast(prob=1.0)
        self.assertEqual(
            repr(transform), ('AutoContrast(prob=1.0, '
                              'level=None, '
                              'min_mag=0.1, '
                              'max_mag=1.9)'))


class TestInvert(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)

    def test_invert(self):
        # test case when no translation is called (prob=0)
        transform = Invert(prob=0.0)
        results_wo_invert = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_invert,
                          self.check_keys)
        # test case when translation is called
        transform = Invert(prob=1.0)
        transform(copy.deepcopy(self.results_mask))

    def test_repr(self):
        transform = Invert(prob=1.0)
        self.assertEqual(
            repr(transform), ('Invert(prob=1.0, '
                              'level=None, '
                              'min_mag=0.1, '
                              'max_mag=1.9)'))
