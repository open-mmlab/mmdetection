# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

from mmdet.datasets.transforms import (AutoAugment, AutoContrast, Brightness,
                                       Color, Contrast, Equalize, Invert,
                                       Posterize, RandAugment, Rotate,
                                       Sharpness, ShearX, ShearY, Solarize,
                                       SolarizeAdd, TranslateX, TranslateY)
from mmdet.utils import register_all_modules
from .utils import check_result_same, construct_toy_data

register_all_modules()


class TestAutoAugment(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map',
                           'homography_matrix')
        self.results_mask = construct_toy_data(poly2mask=True)
        self.img_fill_val = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_autoaugment(self):
        # test AutoAugment equipped with Shear
        policies = [[
            dict(type='ShearX', prob=1.0, level=3, reversal_prob=0.0),
            dict(type='ShearY', prob=1.0, level=7, reversal_prob=1.0)
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_shearx = ShearX(prob=1.0, level=3, reversal_prob=0.0)
        transform_sheary = ShearY(prob=1.0, level=7, reversal_prob=1.0)
        results_sheared = transform_sheary(
            transform_shearx(copy.deepcopy(self.results_mask)))
        check_result_same(results_sheared, results_auto, self.check_keys)

        # test AutoAugment equipped with Rotate
        policies = [[
            dict(type='Rotate', prob=1.0, level=10, reversal_prob=0.0),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_rotate = Rotate(prob=1.0, level=10, reversal_prob=0.0)
        results_rotated = transform_rotate(copy.deepcopy(self.results_mask))
        check_result_same(results_rotated, results_auto, self.check_keys)

        # test AutoAugment equipped with Translate
        policies = [[
            dict(
                type='TranslateX',
                prob=1.0,
                level=10,
                max_mag=1.0,
                reversal_prob=0.0),
            dict(
                type='TranslateY',
                prob=1.0,
                level=10,
                max_mag=1.0,
                reversal_prob=1.0)
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_translatex = TranslateX(
            prob=1.0, level=10, max_mag=1.0, reversal_prob=0.0)
        transform_translatey = TranslateY(
            prob=1.0, level=10, max_mag=1.0, reversal_prob=1.0)
        results_translated = transform_translatey(
            transform_translatex(copy.deepcopy(self.results_mask)))
        check_result_same(results_translated, results_auto, self.check_keys)

        # test AutoAugment equipped with Brightness
        policies = [[
            dict(type='Brightness', prob=1.0, level=3),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_brightness = Brightness(prob=1.0, level=3)
        results_brightness = transform_brightness(
            copy.deepcopy(self.results_mask))
        check_result_same(results_brightness, results_auto, self.check_keys)

        # test AutoAugment equipped with Color
        policies = [[
            dict(type='Color', prob=1.0, level=3),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_color = Color(prob=1.0, level=3)
        results_colored = transform_color(copy.deepcopy(self.results_mask))
        check_result_same(results_colored, results_auto, self.check_keys)

        # test AutoAugment equipped with Contrast
        policies = [[
            dict(type='Contrast', prob=1.0, level=3),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_contrast = Contrast(prob=1.0, level=3)
        results_contrasted = transform_contrast(
            copy.deepcopy(self.results_mask))
        check_result_same(results_contrasted, results_auto, self.check_keys)

        # test AutoAugment equipped with Sharpness
        policies = [[
            dict(type='Sharpness', prob=1.0, level=3),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_sharpness = Sharpness(prob=1.0, level=3)
        results_sharpness = transform_sharpness(
            copy.deepcopy(self.results_mask))
        check_result_same(results_sharpness, results_auto, self.check_keys)

        # test AutoAugment equipped with Solarize
        policies = [[
            dict(type='Solarize', prob=1.0, level=3),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_solarize = Solarize(prob=1.0, level=3)
        results_solarized = transform_solarize(
            copy.deepcopy(self.results_mask))
        check_result_same(results_solarized, results_auto, self.check_keys)

        # test AutoAugment equipped with SolarizeAdd
        policies = [[
            dict(type='SolarizeAdd', prob=1.0, level=3),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_solarizeadd = SolarizeAdd(prob=1.0, level=3)
        results_solarizeadded = transform_solarizeadd(
            copy.deepcopy(self.results_mask))
        check_result_same(results_solarizeadded, results_auto, self.check_keys)

        # test AutoAugment equipped with Posterize
        policies = [[
            dict(type='Posterize', prob=1.0, level=3),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_posterize = Posterize(prob=1.0, level=3)
        results_posterized = transform_posterize(
            copy.deepcopy(self.results_mask))
        check_result_same(results_posterized, results_auto, self.check_keys)

        # test AutoAugment equipped with Equalize
        policies = [[
            dict(type='Equalize', prob=1.0),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_equalize = Equalize(prob=1.0)
        results_equalized = transform_equalize(
            copy.deepcopy(self.results_mask))
        check_result_same(results_equalized, results_auto, self.check_keys)

        # test AutoAugment equipped with AutoContrast
        policies = [[
            dict(type='AutoContrast', prob=1.0),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_autocontrast = AutoContrast(prob=1.0)
        results_autocontrast = transform_autocontrast(
            copy.deepcopy(self.results_mask))
        check_result_same(results_autocontrast, results_auto, self.check_keys)

        # test AutoAugment equipped with Invert
        policies = [[
            dict(type='Invert', prob=1.0),
        ]]
        transform_auto = AutoAugment(policies=policies)
        results_auto = transform_auto(copy.deepcopy(self.results_mask))
        transform_invert = Invert(prob=1.0)
        results_inverted = transform_invert(copy.deepcopy(self.results_mask))
        check_result_same(results_inverted, results_auto, self.check_keys)

        # test AutoAugment equipped with default policies
        transform_auto = AutoAugment()
        transform_auto(copy.deepcopy(self.results_mask))

    def test_repr(self):
        policies = [[
            dict(type='Rotate', prob=1.0, level=10, reversal_prob=0.0),
            dict(type='Invert', prob=1.0),
        ]]
        transform = AutoAugment(policies=policies)
        self.assertEqual(
            repr(transform), ('AutoAugment('
                              'policies=[['
                              "{'type': 'Rotate', 'prob': 1.0, "
                              "'level': 10, 'reversal_prob': 0.0}, "
                              "{'type': 'Invert', 'prob': 1.0}]], "
                              'prob=None)'))


class TestRandAugment(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map',
                           'homography_matrix')
        self.results_mask = construct_toy_data(poly2mask=True)
        self.img_fill_val = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_randaugment(self):
        # test RandAugment equipped with Rotate
        aug_space = [[
            dict(type='Rotate', prob=1.0, level=10, reversal_prob=0.0)
        ]]
        transform_rand = RandAugment(aug_space=aug_space, aug_num=1)
        results_rand = transform_rand(copy.deepcopy(self.results_mask))
        transform_rotate = Rotate(prob=1.0, level=10, reversal_prob=0.0)
        results_rotated = transform_rotate(copy.deepcopy(self.results_mask))
        check_result_same(results_rotated, results_rand, self.check_keys)

        # test RandAugment equipped with default augmentation space
        transform_rand = RandAugment()
        transform_rand(copy.deepcopy(self.results_mask))

    def test_repr(self):
        aug_space = [
            [dict(type='Rotate')],
            [dict(type='Invert')],
        ]
        transform = RandAugment(aug_space=aug_space)
        self.assertEqual(
            repr(transform), ('RandAugment('
                              'aug_space=['
                              "[{'type': 'Rotate'}], "
                              "[{'type': 'Invert'}]], "
                              'aug_num=2, '
                              'prob=None)'))
