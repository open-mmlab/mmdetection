# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np

from mmdet.datasets.transforms import (GeomTransform, Rotate, ShearX, ShearY,
                                       TranslateX, TranslateY)
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from .utils import check_result_same, construct_toy_data


class TestGeomTransform(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)
        self.img_border_value = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_geomtransform(self):
        # test assertion for invalid prob
        with self.assertRaises(AssertionError):
            transform = GeomTransform(
                prob=-0.5, level=1, min_mag=0.0, max_mag=1.0)

        # test assertion for invalid value of level
        with self.assertRaises(AssertionError):
            transform = GeomTransform(
                prob=0.5, level=-1, min_mag=0.0, max_mag=1.0)

        # test assertion for invalid value of min_mag and max_mag
        with self.assertRaises(AssertionError):
            transform = ShearX(prob=0.5, level=2, min_mag=1.0, max_mag=0.0)

        # test assertion for the num of elements in tuple img_border_value
        with self.assertRaises(AssertionError):
            transform = GeomTransform(
                prob=0.5,
                level=1,
                min_mag=0.0,
                max_mag=1.0,
                img_border_value=(128, 128, 128, 128))

        # test ValueError for invalid type of img_border_value
        with self.assertRaises(ValueError):
            transform = GeomTransform(
                prob=0.5,
                level=1,
                min_mag=0.0,
                max_mag=1.0,
                img_border_value=[128, 128, 128])

        # test assertion for invalid value of img_border_value
        with self.assertRaises(AssertionError):
            transform = GeomTransform(
                prob=0.5,
                level=1,
                min_mag=0.0,
                max_mag=1.0,
                img_border_value=(128, -1, 256))

        # test case when no aug (prob=0)
        transform = GeomTransform(
            prob=0.,
            level=10,
            min_mag=0.0,
            max_mag=1.0,
            img_border_value=self.img_border_value)
        results_wo_aug = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_aug, self.check_keys)

    def test_repr(self):
        transform = GeomTransform(
            prob=0.5,
            level=5,
            min_mag=0.0,
            max_mag=1.0,
        )
        self.assertEqual(
            repr(transform), ('GeomTransform(prob=0.5, '
                              'level=5, '
                              'min_mag=0.0, '
                              'max_mag=1.0, '
                              'reversal_prob=0.5, '
                              'img_border_value=(128.0, 128.0, 128.0), '
                              'mask_border_value=0, '
                              'seg_ignore_label=255, '
                              'interpolation=bilinear)'))


class TestShearX(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)
        self.results_poly = construct_toy_data(poly2mask=False)
        self.results_mask_boxtype = construct_toy_data(
            poly2mask=True, use_box_type=True)
        self.img_border_value = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_shearx(self):
        # test assertion for invalid value of min_mag
        with self.assertRaises(AssertionError):
            transform = ShearX(prob=0.5, level=2, min_mag=-30.)
        # test assertion for invalid value of max_mag
        with self.assertRaises(AssertionError):
            transform = ShearX(prob=0.5, level=2, max_mag=100.)

        # test case when no shear horizontally (level=0)
        transform = ShearX(
            prob=1.0,
            level=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label,
        )
        results_wo_shearx = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_shearx,
                          self.check_keys)

        # test shear horizontally, magnitude=-1
        transform = ShearX(
            prob=1.0,
            level=10,
            max_mag=45.,
            reversal_prob=1.0,
            img_border_value=self.img_border_value)
        results_sheared = transform(copy.deepcopy(self.results_mask))
        results_gt = copy.deepcopy(self.results_mask)
        img_gt = np.array([[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 9, 10]],
                          dtype=np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        img_gt[1, 0, :] = np.array(self.img_border_value)
        img_gt[2, 0, :] = np.array(self.img_border_value)
        img_gt[2, 1, :] = np.array(self.img_border_value)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = np.array([[1, 0, 4, 2]], dtype=np.float32)
        results_gt['gt_bboxes_labels'] = np.array([13], dtype=np.int64)
        gt_masks = np.array([[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 13, 255, 255], [255, 255, 13, 13], [255, 255, 255, 13]],
            dtype=self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_sheared, self.check_keys)

        # test PolygonMasks with shear horizontally, magnitude=1
        results_sheared = transform(copy.deepcopy(self.results_poly))
        gt_masks = [[np.array([3, 2, 1, 0, 3, 1], dtype=np.float32)]]
        results_gt['gt_masks'] = PolygonMasks(gt_masks, 3, 4)
        check_result_same(results_gt, results_sheared, self.check_keys)

    def test_shearx_use_box_type(self):
        # test case when no shear horizontally (level=0)
        transform = ShearX(
            prob=1.0,
            level=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label,
        )
        results_wo_shearx = transform(copy.deepcopy(self.results_mask_boxtype))
        check_result_same(self.results_mask_boxtype, results_wo_shearx,
                          self.check_keys)

        # test shear horizontally, magnitude=-1
        transform = ShearX(
            prob=1.0,
            level=10,
            max_mag=45.,
            reversal_prob=1.0,
            img_border_value=self.img_border_value)
        results_sheared = transform(copy.deepcopy(self.results_mask_boxtype))
        results_gt = copy.deepcopy(self.results_mask_boxtype)
        img_gt = np.array([[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 9, 10]],
                          dtype=np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        img_gt[1, 0, :] = np.array(self.img_border_value)
        img_gt[2, 0, :] = np.array(self.img_border_value)
        img_gt[2, 1, :] = np.array(self.img_border_value)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = HorizontalBoxes(
            np.array([[1, 0, 4, 2]], dtype=np.float32))
        results_gt['gt_bboxes_labels'] = np.array([13], dtype=np.int64)
        gt_masks = np.array([[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 13, 255, 255], [255, 255, 13, 13], [255, 255, 255, 13]],
            dtype=self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_sheared, self.check_keys)

    def test_repr(self):
        transform = ShearX(prob=0.5, level=10)
        self.assertEqual(
            repr(transform), ('ShearX(prob=0.5, '
                              'level=10, '
                              'min_mag=0.0, '
                              'max_mag=30.0, '
                              'reversal_prob=0.5, '
                              'img_border_value=(128.0, 128.0, 128.0), '
                              'mask_border_value=0, '
                              'seg_ignore_label=255, '
                              'interpolation=bilinear)'))


class TestShearY(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)
        self.results_poly = construct_toy_data(poly2mask=False)
        self.results_mask_boxtype = construct_toy_data(
            poly2mask=True, use_box_type=True)
        self.img_border_value = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_sheary(self):
        # test assertion for invalid value of min_mag
        with self.assertRaises(AssertionError):
            transform = ShearY(prob=0.5, level=2, min_mag=-30.)
        # test assertion for invalid value of max_mag
        with self.assertRaises(AssertionError):
            transform = ShearY(prob=0.5, level=2, max_mag=100.)

        # test case when no shear vertically (level=0)
        transform = ShearY(
            prob=1.0,
            level=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label,
        )
        results_wo_sheary = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_sheary,
                          self.check_keys)

        # test shear vertically, magnitude=1
        transform = ShearY(prob=1., level=10, max_mag=45., reversal_prob=0.)
        results_sheared = transform(copy.deepcopy(self.results_mask))
        results_gt = copy.deepcopy(self.results_mask)
        img_gt = np.array(
            [[1, 6, 11, 128], [5, 10, 128, 128], [9, 128, 128, 128]],
            dtype=np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = np.array([[1, 0, 2, 1]], dtype=np.float32)
        results_gt['gt_bboxes_labels'] = np.array([13], dtype=np.int64)
        gt_masks = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 13, 255, 255], [255, 13, 255, 255], [255, 255, 255, 255]],
            dtype=self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_sheared, self.check_keys)

        # test PolygonMasks with shear vertically, magnitude=-1
        results_sheared = transform(copy.deepcopy(self.results_poly))
        gt_masks = [[np.array([1, 1, 1, 0, 2, 0], dtype=np.float32)]]
        results_gt['gt_masks'] = PolygonMasks(gt_masks, 3, 4)
        check_result_same(results_gt, results_sheared, self.check_keys)

    def test_sheary_use_box_type(self):
        # test case when no shear vertically (level=0)
        transform = ShearY(
            prob=1.0,
            level=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label,
        )
        results_wo_sheary = transform(copy.deepcopy(self.results_mask_boxtype))
        check_result_same(self.results_mask_boxtype, results_wo_sheary,
                          self.check_keys)

        # test shear vertically, magnitude=1
        transform = ShearY(prob=1., level=10, max_mag=45., reversal_prob=0.)
        results_sheared = transform(copy.deepcopy(self.results_mask_boxtype))
        results_gt = copy.deepcopy(self.results_mask_boxtype)
        img_gt = np.array(
            [[1, 6, 11, 128], [5, 10, 128, 128], [9, 128, 128, 128]],
            dtype=np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = HorizontalBoxes(
            np.array([[1, 0, 2, 1]], dtype=np.float32))
        results_gt['gt_bboxes_labels'] = np.array([13], dtype=np.int64)
        gt_masks = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 13, 255, 255], [255, 13, 255, 255], [255, 255, 255, 255]],
            dtype=self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_sheared, self.check_keys)

    def test_repr(self):
        transform = ShearY(prob=0.5, level=10)
        self.assertEqual(
            repr(transform), ('ShearY(prob=0.5, '
                              'level=10, '
                              'min_mag=0.0, '
                              'max_mag=30.0, '
                              'reversal_prob=0.5, '
                              'img_border_value=(128.0, 128.0, 128.0), '
                              'mask_border_value=0, '
                              'seg_ignore_label=255, '
                              'interpolation=bilinear)'))


class TestRotate(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)
        self.results_poly = construct_toy_data(poly2mask=False)
        self.results_mask_boxtype = construct_toy_data(
            poly2mask=True, use_box_type=True)
        self.img_border_value = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_rotate(self):
        # test assertion for invalid value of min_mag
        with self.assertRaises(AssertionError):
            transform = ShearY(prob=0.5, level=2, min_mag=-90.0)
        # test assertion for invalid value of max_mag
        with self.assertRaises(AssertionError):
            transform = ShearY(prob=0.5, level=2, max_mag=270.0)

        # test case when no rotate aug (level=0)
        transform = Rotate(
            prob=1.,
            level=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label,
        )
        results_wo_rotate = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_rotate,
                          self.check_keys)

        # test clockwise rotation with angle 90
        transform = Rotate(
            prob=1.,
            level=10,
            max_mag=90.0,
            # set reversal_prob to 1 for clockwise rotation
            reversal_prob=1.,
        )
        results_rotated = transform(copy.deepcopy(self.results_mask))
        # The image, masks, and semantic segmentation map
        # will be bilinearly interpolated.
        img_gt = np.array([[69, 8, 4, 65], [69, 9, 5, 65],
                           [70, 10, 6, 66]]).astype(np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        results_gt = copy.deepcopy(self.results_mask)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = np.array([[0.5, 0.5, 2.5, 1.5]],
                                           dtype=np.float32)
        gt_masks = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 13, 13, 13], [255, 255, 13, 255],
             [255, 255, 255,
              255]]).astype(self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_rotated, self.check_keys)

        # test clockwise rotation with angle 90, PolygonMasks
        results_rotated = transform(copy.deepcopy(self.results_poly))
        gt_masks = [[np.array([0, 1, 0, 1, 0, 2], dtype=np.float)]]
        results_gt['gt_masks'] = PolygonMasks(gt_masks, 3, 4)
        check_result_same(results_gt, results_rotated, self.check_keys)

        # test counter-clockwise rotation with angle 90
        transform = Rotate(
            prob=1.0,
            level=10,
            max_mag=90.0,
            # set reversal_prob to 0 for counter-clockwise rotation
            reversal_prob=0.0,
        )
        results_rotated = transform(copy.deepcopy(self.results_mask))
        # The image, masks, and  semantic segmentation map
        # will be bilinearly interpolated.
        img_gt = np.array([[66, 6, 10, 70], [65, 5, 9, 69],
                           [65, 4, 8, 69]]).astype(np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        results_gt = copy.deepcopy(self.results_mask)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = np.array([[0.5, 0.5, 2.5, 1.5]],
                                           dtype=np.float32)
        gt_masks = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 255, 255, 255], [255, 13, 255, 255],
             [13, 13, 13, 255]]).astype(self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_rotated, self.check_keys)

        # test counter-clockwise rotation with angle 90, PolygonMasks
        results_rotated = transform(copy.deepcopy(self.results_poly))
        gt_masks = [[np.array([2, 0, 0, 0, 1, 0], dtype=np.float)]]
        results_gt['gt_masks'] = PolygonMasks(gt_masks, 3, 4)
        check_result_same(results_gt, results_rotated, self.check_keys)

    def test_rotate_use_box_type(self):
        # test case when no rotate aug (level=0)
        transform = Rotate(
            prob=1.,
            level=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label,
        )
        results_wo_rotate = transform(copy.deepcopy(self.results_mask_boxtype))
        check_result_same(self.results_mask_boxtype, results_wo_rotate,
                          self.check_keys)

        # test clockwise rotation with angle 90
        transform = Rotate(
            prob=1.,
            level=10,
            max_mag=90.0,
            # set reversal_prob to 1 for clockwise rotation
            reversal_prob=1.,
        )
        results_rotated = transform(copy.deepcopy(self.results_mask_boxtype))
        # The image, masks, and semantic segmentation map
        # will be bilinearly interpolated.
        img_gt = np.array([[69, 8, 4, 65], [69, 9, 5, 65],
                           [70, 10, 6, 66]]).astype(np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        results_gt = copy.deepcopy(self.results_mask_boxtype)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = HorizontalBoxes(
            np.array([[0.5, 0.5, 2.5, 1.5]], dtype=np.float32))
        gt_masks = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 13, 13, 13], [255, 255, 13, 255],
             [255, 255, 255,
              255]]).astype(self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_rotated, self.check_keys)

        # test counter-clockwise rotation with angle 90
        transform = Rotate(
            prob=1.0,
            level=10,
            max_mag=90.0,
            # set reversal_prob to 0 for counter-clockwise rotation
            reversal_prob=0.0,
        )
        results_rotated = transform(copy.deepcopy(self.results_mask_boxtype))
        # The image, masks, and  semantic segmentation map
        # will be bilinearly interpolated.
        img_gt = np.array([[66, 6, 10, 70], [65, 5, 9, 69],
                           [65, 4, 8, 69]]).astype(np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        results_gt = copy.deepcopy(self.results_mask_boxtype)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = HorizontalBoxes(
            np.array([[0.5, 0.5, 2.5, 1.5]], dtype=np.float32))
        gt_masks = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 255, 255, 255], [255, 13, 255, 255],
             [13, 13, 13, 255]]).astype(self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_rotated, self.check_keys)

    def test_repr(self):
        transform = Rotate(prob=0.5, level=5)
        self.assertEqual(
            repr(transform), ('Rotate(prob=0.5, '
                              'level=5, '
                              'min_mag=0.0, '
                              'max_mag=30.0, '
                              'reversal_prob=0.5, '
                              'img_border_value=(128.0, 128.0, 128.0), '
                              'mask_border_value=0, '
                              'seg_ignore_label=255, '
                              'interpolation=bilinear)'))


class TestTranslateX(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)
        self.results_poly = construct_toy_data(poly2mask=False)
        self.results_mask_boxtype = construct_toy_data(
            poly2mask=True, use_box_type=True)
        self.img_border_value = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_translatex(self):
        # test assertion for invalid value of min_mag
        with self.assertRaises(AssertionError):
            transform = TranslateX(prob=0.5, level=2, min_mag=-1.)
        # test assertion for invalid value of max_mag
        with self.assertRaises(AssertionError):
            transform = TranslateX(prob=0.5, level=2, max_mag=1.1)

        # test case when level=0 (without translate aug)
        transform = TranslateX(
            prob=1.0,
            level=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label)
        results_wo_translatex = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_translatex,
                          self.check_keys)

        # test translate horizontally, magnitude=-1
        transform = TranslateX(
            prob=1.0,
            level=10,
            max_mag=0.3,
            reversal_prob=0.0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label)
        results_translated = transform(copy.deepcopy(self.results_mask))
        img_gt = np.array([[2, 3, 4, 0], [6, 7, 8, 0], [10, 11, 12,
                                                        0]]).astype(np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        img_gt[:, 3, :] = np.array(self.img_border_value)
        results_gt = copy.deepcopy(self.results_mask)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = np.array([[0, 0, 1, 2]], dtype=np.float32)
        gt_masks = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[13, 255, 255, 255], [13, 13, 255, 255],
             [13, 255, 255,
              255]]).astype(self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_translated, self.check_keys)

        # test PolygonMasks with translate horizontally.
        results_translated = transform(copy.deepcopy(self.results_poly))
        gt_masks = [[np.array([0, 2, 0, 0, 1, 1], dtype=np.float32)]]
        results_gt['gt_masks'] = PolygonMasks(gt_masks, 3, 4)
        check_result_same(results_gt, results_translated, self.check_keys)

    def test_translatex_use_box_type(self):
        # test case when level=0 (without translate aug)
        transform = TranslateX(
            prob=1.0,
            level=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label)
        results_wo_translatex = transform(
            copy.deepcopy(self.results_mask_boxtype))
        check_result_same(self.results_mask_boxtype, results_wo_translatex,
                          self.check_keys)

        # test translate horizontally, magnitude=-1
        transform = TranslateX(
            prob=1.0,
            level=10,
            max_mag=0.3,
            reversal_prob=0.0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label)
        results_translated = transform(
            copy.deepcopy(self.results_mask_boxtype))
        img_gt = np.array([[2, 3, 4, 0], [6, 7, 8, 0], [10, 11, 12,
                                                        0]]).astype(np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        img_gt[:, 3, :] = np.array(self.img_border_value)
        results_gt = copy.deepcopy(self.results_mask)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = HorizontalBoxes(
            np.array([[0, 0, 1, 2]], dtype=np.float32))
        gt_masks = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[13, 255, 255, 255], [13, 13, 255, 255],
             [13, 255, 255,
              255]]).astype(self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_translated, self.check_keys)

    def test_repr(self):
        transform = TranslateX(prob=0.5, level=5)
        self.assertEqual(
            repr(transform), ('TranslateX(prob=0.5, '
                              'level=5, '
                              'min_mag=0.0, '
                              'max_mag=0.1, '
                              'reversal_prob=0.5, '
                              'img_border_value=(128.0, 128.0, 128.0), '
                              'mask_border_value=0, '
                              'seg_ignore_label=255, '
                              'interpolation=bilinear)'))


class TestTranslateY(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)
        self.results_poly = construct_toy_data(poly2mask=False)
        self.results_mask_boxtype = construct_toy_data(
            poly2mask=True, use_box_type=True)
        self.img_border_value = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_translatey(self):
        # test assertion for invalid value of min_mag
        with self.assertRaises(AssertionError):
            transform = TranslateY(prob=0.5, level=2, min_mag=-1.0)
        # test assertion for invalid value of max_mag
        with self.assertRaises(AssertionError):
            transform = TranslateY(prob=0.5, level=2, max_mag=1.1)

        # test case when level=0 (without translate aug)
        transform = TranslateY(
            prob=1.0,
            level=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label)
        results_wo_translatey = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_translatey,
                          self.check_keys)

        # test translate vertically, magnitude=-1
        transform = TranslateY(
            prob=1.0,
            level=10,
            max_mag=0.4,
            reversal_prob=0.0,
            seg_ignore_label=self.seg_ignore_label)

        results_translated = transform(copy.deepcopy(self.results_mask))
        img_gt = np.array([[5, 6, 7, 8], [9, 10, 11, 12],
                           [128, 128, 128, 128]]).astype(np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        results_gt = copy.deepcopy(self.results_mask)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = np.array([[1, 0, 2, 1]], dtype=np.float32)
        gt_masks = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 13, 13, 255], [255, 13, 255, 255],
             [255, 255, 255,
              255]]).astype(self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_translated, self.check_keys)

        # test PolygonMasks with translate vertically.
        results_translated = transform(copy.deepcopy(self.results_poly))
        gt_masks = [[np.array([1, 1, 1, 0, 2, 0], dtype=np.float32)]]
        results_gt['gt_masks'] = PolygonMasks(gt_masks, 3, 4)
        check_result_same(results_gt, results_translated, self.check_keys)

    def test_translatey_use_box_type(self):
        # test case when level=0 (without translate aug)
        transform = TranslateY(
            prob=1.0,
            level=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label)
        results_wo_translatey = transform(
            copy.deepcopy(self.results_mask_boxtype))
        check_result_same(self.results_mask_boxtype, results_wo_translatey,
                          self.check_keys)

        # test translate vertically, magnitude=-1
        transform = TranslateY(
            prob=1.0,
            level=10,
            max_mag=0.4,
            reversal_prob=0.0,
            seg_ignore_label=self.seg_ignore_label)

        results_translated = transform(
            copy.deepcopy(self.results_mask_boxtype))
        img_gt = np.array([[5, 6, 7, 8], [9, 10, 11, 12],
                           [128, 128, 128, 128]]).astype(np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        results_gt = copy.deepcopy(self.results_mask_boxtype)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = HorizontalBoxes(
            np.array([[1, 0, 2, 1]], dtype=np.float32))
        gt_masks = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 13, 13, 255], [255, 13, 255, 255],
             [255, 255, 255,
              255]]).astype(self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_translated, self.check_keys)

    def test_repr(self):
        transform = TranslateX(prob=0.5, level=5)
        self.assertEqual(
            repr(transform), ('TranslateX(prob=0.5, '
                              'level=5, '
                              'min_mag=0.0, '
                              'max_mag=0.1, '
                              'reversal_prob=0.5, '
                              'img_border_value=(128.0, 128.0, 128.0), '
                              'mask_border_value=0, '
                              'seg_ignore_label=255, '
                              'interpolation=bilinear)'))
