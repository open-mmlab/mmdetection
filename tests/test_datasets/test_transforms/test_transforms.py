# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import mmcv
import numpy as np
import torch
from mmcv.transforms import LoadImageFromFile

# yapf:disable
from mmdet.datasets.transforms import (CopyPaste, CutOut, Expand,
                                       FixShapeResize, MinIoURandomCrop, MixUp,
                                       Mosaic, Pad, PhotoMetricDistortion,
                                       RandomAffine, RandomCenterCropPad,
                                       RandomCrop, RandomErasing, RandomFlip,
                                       RandomShift, Resize, SegRescale,
                                       YOLOXHSVRandomAug)
# yapf:enable
from mmdet.evaluation import bbox_overlaps
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, bbox_project
from mmdet.structures.mask import BitmapMasks
from .utils import construct_toy_data, create_full_masks, create_random_bboxes

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None
# yapf:enable


class TestResize(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod()
        -> tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.data_info1 = dict(
            img=np.random.random((1333, 800, 3)),
            gt_seg_map=np.random.random((1333, 800, 3)),
            gt_bboxes=np.array([[0, 0, 112, 112]], dtype=np.float32),
            gt_masks=BitmapMasks(
                rng.rand(1, 1333, 800), height=1333, width=800))
        self.data_info2 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[200, 150, 600, 450]], dtype=np.float32),
            dtype=np.float32)
        self.data_info3 = dict(img=np.random.random((300, 400, 3)))

    def test_resize(self):
        # test keep_ratio is True
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (2000, 1200))
        self.assertEqual(results['scale_factor'], (1200 / 800, 2000 / 1333))

        # test resize_bboxes/seg/masks
        transform = Resize(scale_factor=(1.5, 2))
        results = transform(copy.deepcopy(self.data_info1))
        self.assertTrue((results['gt_bboxes'] == np.array([[0, 0, 168,
                                                            224]])).all())
        self.assertEqual(results['gt_masks'].height, 2666)
        self.assertEqual(results['gt_masks'].width, 1200)
        self.assertEqual(results['gt_seg_map'].shape[:2], (2666, 1200))

        # test clip_object_border = False
        transform = Resize(scale=(200, 150), clip_object_border=False)
        results = transform(self.data_info2)
        self.assertTrue((results['gt_bboxes'] == np.array([100, 75, 300,
                                                           225])).all())

        # test only with image
        transform = Resize(scale=(200, 150), clip_object_border=False)
        results = transform(self.data_info3)
        self.assertTupleEqual(results['img'].shape[:2], (150, 200))

        # test geometric transformation with homography matrix
        transform = Resize(scale_factor=(1.5, 2))
        results = transform(copy.deepcopy(self.data_info1))
        self.assertTrue((bbox_project(
            copy.deepcopy(self.data_info1['gt_bboxes']),
            results['homography_matrix']) == results['gt_bboxes']).all())

    def test_resize_use_box_type(self):
        data_info1 = copy.deepcopy(self.data_info1)
        data_info1['gt_bboxes'] = HorizontalBoxes(data_info1['gt_bboxes'])
        data_info2 = copy.deepcopy(self.data_info2)
        data_info2['gt_bboxes'] = HorizontalBoxes(data_info2['gt_bboxes'])
        # test keep_ratio is True
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        results = transform(copy.deepcopy(data_info1))
        self.assertEqual(results['img_shape'], (2000, 1200))
        self.assertEqual(results['scale_factor'], (1200 / 800, 2000 / 1333))

        # test resize_bboxes/seg/masks
        transform = Resize(scale_factor=(1.5, 2))
        results = transform(copy.deepcopy(data_info1))
        self.assertTrue(
            (results['gt_bboxes'].numpy() == np.array([[0, 0, 168,
                                                        224]])).all())
        self.assertEqual(results['gt_masks'].height, 2666)
        self.assertEqual(results['gt_masks'].width, 1200)
        self.assertEqual(results['gt_seg_map'].shape[:2], (2666, 1200))

        # test clip_object_border = False
        transform = Resize(scale=(200, 150), clip_object_border=False)
        results = transform(data_info2)
        self.assertTrue(
            (results['gt_bboxes'].numpy() == np.array([100, 75, 300,
                                                       225])).all())

        # test geometric transformation with homography matrix
        transform = Resize(scale_factor=(1.5, 2))
        results = transform(copy.deepcopy(data_info1))
        self.assertTrue((bbox_project(
            copy.deepcopy(data_info1['gt_bboxes'].numpy()),
            results['homography_matrix']) == results['gt_bboxes'].numpy()
                         ).all())

    def test_repr(self):
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        self.assertEqual(
            repr(transform), ('Resize(scale=(2000, 2000), '
                              'scale_factor=None, keep_ratio=True, '
                              'clip_object_border=True), backend=cv2), '
                              'interpolation=bilinear)'))


class TestFIXShapeResize(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.data_info1 = dict(
            img=np.random.random((1333, 800, 3)),
            gt_seg_map=np.random.random((1333, 800, 3)),
            gt_bboxes=np.array([[0, 0, 112, 1333]], dtype=np.float32),
            gt_masks=BitmapMasks(
                rng.rand(1, 1333, 800), height=1333, width=800))
        self.data_info2 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[200, 150, 600, 450]], dtype=np.float32),
            dtype=np.float32)
        self.data_info3 = dict(img=np.random.random((300, 400, 3)))
        self.data_info4 = dict(
            img=np.random.random((600, 800, 3)),
            gt_bboxes=np.array([[200, 150, 300, 400]], dtype=np.float32),
            dtype=np.float32)

    def test_resize(self):
        # test keep_ratio is True
        transform = FixShapeResize(width=2000, height=800, keep_ratio=True)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (800, 2000))
        self.assertEqual(results['scale_factor'], (800 / 1333, 800 / 1333))
        # test resize_bboxes/seg/masks
        transform = FixShapeResize(width=2000, height=800, keep_ratio=False)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertTrue((results['gt_bboxes'] == np.array([[0, 0, 280,
                                                            800]])).all())
        self.assertEqual(results['gt_masks'].height, 800)
        self.assertEqual(results['gt_masks'].width, 2000)
        self.assertEqual(results['gt_seg_map'].shape[:2], (800, 2000))

        # test clip_object_border = False
        transform = FixShapeResize(
            width=200, height=150, clip_object_border=False)
        results = transform(copy.deepcopy(self.data_info2))
        self.assertTrue((results['gt_bboxes'] == np.array([100, 75, 300,
                                                           225])).all())

        # test only with image
        transform = FixShapeResize(
            width=200, height=150, clip_object_border=False)
        results = transform(self.data_info3)
        self.assertTupleEqual(results['img'].shape[:2], (150, 200))

        # test geometric transformation with homography matrix
        transform = FixShapeResize(width=400, height=300)
        results = transform(copy.deepcopy(self.data_info4))
        self.assertTrue((bbox_project(
            copy.deepcopy(self.data_info4['gt_bboxes']),
            results['homography_matrix']) == results['gt_bboxes']).all())

    def test_resize_with_boxlist(self):
        data_info1 = copy.deepcopy(self.data_info1)
        data_info1['gt_bboxes'] = HorizontalBoxes(data_info1['gt_bboxes'])
        data_info2 = copy.deepcopy(self.data_info2)
        data_info2['gt_bboxes'] = HorizontalBoxes(data_info2['gt_bboxes'])
        data_info4 = copy.deepcopy(self.data_info4)
        data_info4['gt_bboxes'] = HorizontalBoxes(data_info4['gt_bboxes'])
        # test keep_ratio is True
        transform = FixShapeResize(width=2000, height=800, keep_ratio=True)
        results = transform(copy.deepcopy(data_info1))
        self.assertEqual(results['img_shape'], (800, 2000))
        self.assertEqual(results['scale_factor'], (800 / 1333, 800 / 1333))

        # test resize_bboxes/seg/masks
        transform = FixShapeResize(width=2000, height=800, keep_ratio=False)
        results = transform(copy.deepcopy(data_info1))
        self.assertTrue(
            (results['gt_bboxes'].numpy() == np.array([[0, 0, 280,
                                                        800]])).all())
        self.assertEqual(results['gt_masks'].height, 800)
        self.assertEqual(results['gt_masks'].width, 2000)
        self.assertEqual(results['gt_seg_map'].shape[:2], (800, 2000))

        # test clip_object_border = False
        transform = FixShapeResize(
            width=200, height=150, clip_object_border=False)
        results = transform(copy.deepcopy(data_info2))
        self.assertTrue(
            (results['gt_bboxes'].numpy() == np.array([100, 75, 300,
                                                       225])).all())

        # test only with image
        transform = FixShapeResize(
            width=200, height=150, clip_object_border=False)
        results = transform(self.data_info3)
        self.assertTupleEqual(results['img'].shape[:2], (150, 200))

        # test geometric transformation with homography matrix
        transform = FixShapeResize(width=400, height=300)
        results = transform(copy.deepcopy(data_info4))
        self.assertTrue((bbox_project(
            copy.deepcopy(self.data_info4['gt_bboxes']),
            results['homography_matrix']) == results['gt_bboxes'].numpy()
                         ).all())

    def test_repr(self):
        transform = FixShapeResize(width=2000, height=2000, keep_ratio=True)
        self.assertEqual(
            repr(transform), ('FixShapeResize(width=2000, height=2000, '
                              'keep_ratio=True, '
                              'clip_object_border=True), backend=cv2), '
                              'interpolation=bilinear)'))


class TestRandomFlip(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results1 = {
            'img': np.random.random((224, 224, 3)),
            'gt_bboxes': np.array([[0, 1, 100, 101]], dtype=np.float32),
            'gt_masks':
            BitmapMasks(rng.rand(1, 224, 224), height=224, width=224),
            'gt_seg_map': np.random.random((224, 224))
        }

        self.results2 = {'img': self.results1['img']}

    def test_transform(self):
        # test with image, gt_bboxes, gt_masks, gt_seg_map
        transform = RandomFlip(1.0)
        results_update = transform.transform(copy.deepcopy(self.results1))
        self.assertTrue(
            (results_update['gt_bboxes'] == np.array([[124, 1, 224,
                                                       101]])).all())
        # test only with image
        transform = RandomFlip(1.0)
        results_update = transform.transform(copy.deepcopy(self.results2))
        self.assertTrue(
            (results_update['img'] == self.results2['img'][:, ::-1]).all())

        # test geometric transformation with homography matrix
        # (1) Horizontal Flip
        transform = RandomFlip(1.0)
        results_update = transform.transform(copy.deepcopy(self.results1))
        bboxes = copy.deepcopy(self.results1['gt_bboxes'])
        self.assertTrue((bbox_project(
            bboxes,
            results_update['homography_matrix']) == results_update['gt_bboxes']
                         ).all())
        # (2) Vertical Flip
        transform = RandomFlip(1.0, direction='vertical')
        results_update = transform.transform(copy.deepcopy(self.results1))
        bboxes = copy.deepcopy(self.results1['gt_bboxes'])
        self.assertTrue((bbox_project(
            bboxes,
            results_update['homography_matrix']) == results_update['gt_bboxes']
                         ).all())
        # (3) Diagonal Flip
        transform = RandomFlip(1.0, direction='diagonal')
        results_update = transform.transform(copy.deepcopy(self.results1))
        bboxes = copy.deepcopy(self.results1['gt_bboxes'])
        self.assertTrue((bbox_project(
            bboxes,
            results_update['homography_matrix']) == results_update['gt_bboxes']
                         ).all())

    def test_transform_use_box_type(self):
        results1 = copy.deepcopy(self.results1)
        results1['gt_bboxes'] = HorizontalBoxes(results1['gt_bboxes'])
        # test with image, gt_bboxes, gt_masks, gt_seg_map
        transform = RandomFlip(1.0)
        results_update = transform.transform(copy.deepcopy(results1))
        self.assertTrue((results_update['gt_bboxes'].numpy() == np.array(
            [[124, 1, 224, 101]])).all())

        # test geometric transformation with homography matrix
        # (1) Horizontal Flip
        transform = RandomFlip(1.0)
        results_update = transform.transform(copy.deepcopy(results1))
        bboxes = copy.deepcopy(results1['gt_bboxes'].numpy())
        self.assertTrue((bbox_project(bboxes,
                                      results_update['homography_matrix']) ==
                         results_update['gt_bboxes'].numpy()).all())
        # (2) Vertical Flip
        transform = RandomFlip(1.0, direction='vertical')
        results_update = transform.transform(copy.deepcopy(results1))
        bboxes = copy.deepcopy(results1['gt_bboxes'].numpy())
        self.assertTrue((bbox_project(bboxes,
                                      results_update['homography_matrix']) ==
                         results_update['gt_bboxes'].numpy()).all())
        # (3) Diagonal Flip
        transform = RandomFlip(1.0, direction='diagonal')
        results_update = transform.transform(copy.deepcopy(results1))
        bboxes = copy.deepcopy(results1['gt_bboxes'].numpy())
        self.assertTrue((bbox_project(bboxes,
                                      results_update['homography_matrix']) ==
                         results_update['gt_bboxes'].numpy()).all())

    def test_repr(self):
        transform = RandomFlip(0.1)
        transform_str = str(transform)
        self.assertIsInstance(transform_str, str)


class TestPad(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results = {
            'img': np.random.random((1333, 800, 3)),
            'gt_masks':
            BitmapMasks(rng.rand(4, 1333, 800), height=1333, width=800)
        }

    def test_transform(self):
        # test pad img/gt_masks with size
        transform = Pad(size=(1200, 2000))
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['img'].shape[:2], (2000, 1200))
        self.assertEqual(results['gt_masks'].masks.shape[1:], (2000, 1200))

        # test pad img/gt_masks with size_divisor
        transform = Pad(size_divisor=11)
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['img'].shape[:2], (1342, 803))
        self.assertEqual(results['gt_masks'].masks.shape[1:], (1342, 803))

        # test pad img/gt_masks with pad_to_square
        transform = Pad(pad_to_square=True)
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['img'].shape[:2], (1333, 1333))
        self.assertEqual(results['gt_masks'].masks.shape[1:], (1333, 1333))

        # test pad img/gt_masks with pad_to_square and size_divisor
        transform = Pad(pad_to_square=True, size_divisor=11)
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['img'].shape[:2], (1342, 1342))
        self.assertEqual(results['gt_masks'].masks.shape[1:], (1342, 1342))

        # test pad img/gt_masks with pad_to_square and size_divisor
        transform = Pad(pad_to_square=True, size_divisor=11)
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['img'].shape[:2], (1342, 1342))
        self.assertEqual(results['gt_masks'].masks.shape[1:], (1342, 1342))

    def test_repr(self):
        transform = Pad(
            pad_to_square=True, size_divisor=11, padding_mode='edge')
        self.assertEqual(
            repr(transform),
            ('Pad(size=None, size_divisor=11, pad_to_square=True, '
             "pad_val={'img': 0, 'seg': 255}), padding_mode=edge)"))


class TestMinIoURandomCrop(unittest.TestCase):

    def test_transform(self):
        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        gt_bboxes = create_random_bboxes(1, results['img_shape'][1],
                                         results['img_shape'][0])
        results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
        results['gt_bboxes'] = gt_bboxes
        transform = MinIoURandomCrop()
        results = transform.transform(copy.deepcopy(results))

        self.assertEqual(results['gt_labels'].shape[0],
                         results['gt_bboxes'].shape[0])
        self.assertEqual(results['gt_labels'].dtype, np.int64)
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)
        self.assertEqual(results['img_shape'], results['img'].shape[:2])

        patch = np.array(
            [0, 0, results['img_shape'][1], results['img_shape'][0]])
        ious = bbox_overlaps(patch.reshape(-1, 4),
                             results['gt_bboxes']).reshape(-1)
        mode = transform.mode
        if mode == 1:
            self.assertTrue(np.equal(results['gt_bboxes'], gt_bboxes).all())
        else:
            self.assertTrue((ious >= mode).all())

    def test_transform_use_box_type(self):
        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        gt_bboxes = create_random_bboxes(1, results['img_shape'][1],
                                         results['img_shape'][0])
        results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
        results['gt_bboxes'] = HorizontalBoxes(gt_bboxes)
        transform = MinIoURandomCrop()
        results = transform.transform(copy.deepcopy(results))

        self.assertEqual(results['gt_labels'].shape[0],
                         results['gt_bboxes'].shape[0])
        self.assertEqual(results['gt_labels'].dtype, np.int64)
        self.assertEqual(results['gt_bboxes'].dtype, torch.float32)

        patch = np.array(
            [0, 0, results['img_shape'][1], results['img_shape'][0]])
        ious = bbox_overlaps(
            patch.reshape(-1, 4), results['gt_bboxes'].numpy()).reshape(-1)
        mode = transform.mode
        if mode == 1:
            self.assertTrue((results['gt_bboxes'].numpy() == gt_bboxes).all())
        else:
            self.assertTrue((ious >= mode).all())

    def test_repr(self):
        transform = MinIoURandomCrop()
        self.assertEqual(
            repr(transform), ('MinIoURandomCrop'
                              '(min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), '
                              'min_crop_size=0.3, '
                              'bbox_clip_border=True)'))


class TestPhotoMetricDistortion(unittest.TestCase):

    def test_transform(self):
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        transform = PhotoMetricDistortion()

        # test uint8 input
        results = dict()
        results['img'] = img
        results = transform.transform(copy.deepcopy(results))
        self.assertEqual(results['img'].dtype, np.float32)

        # test float32 input
        results = dict()
        results['img'] = img.astype(np.float32)
        results = transform.transform(copy.deepcopy(results))
        self.assertEqual(results['img'].dtype, np.float32)

    def test_repr(self):
        transform = PhotoMetricDistortion()
        self.assertEqual(
            repr(transform), ('PhotoMetricDistortion'
                              '(brightness_delta=32, '
                              'contrast_range=(0.5, 1.5), '
                              'saturation_range=(0.5, 1.5), '
                              'hue_delta=18)'))


class TestExpand(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results = {
            'img': np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes': np.array([[0, 1, 100, 101]]),
            'gt_masks':
            BitmapMasks(rng.rand(1, 224, 224), height=224, width=224),
            'gt_seg_map': np.random.random((224, 224))
        }

    def test_transform(self):
        transform = Expand()
        results = transform.transform(copy.deepcopy(self.results))
        self.assertEqual(results['img_shape'], results['img'].shape[:2])
        self.assertEqual(
            results['img_shape'],
            (results['gt_masks'].height, results['gt_masks'].width))
        self.assertEqual(results['img_shape'], results['gt_seg_map'].shape)

    def test_transform_use_box_type(self):
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])
        transform = Expand()
        results = transform.transform(results)
        self.assertEqual(
            results['img_shape'],
            (results['gt_masks'].height, results['gt_masks'].width))
        self.assertEqual(results['img_shape'], results['gt_seg_map'].shape)

    def test_repr(self):
        transform = Expand()
        self.assertEqual(
            repr(transform), ('Expand'
                              '(mean=(0, 0, 0), to_rgb=True, '
                              'ratio_range=(1, 4), '
                              'seg_ignore_label=None, '
                              'prob=0.5)'))


class TestSegRescale(unittest.TestCase):

    def setUp(self) -> None:
        seg_map = np.random.randint(0, 255, size=(32, 32), dtype=np.int32)
        self.results = {'gt_seg_map': seg_map}

    def test_transform(self):
        # test scale_factor != 1
        transform = SegRescale(scale_factor=2)
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['gt_seg_map'].shape[:2], (64, 64))
        # test scale_factor = 1
        transform = SegRescale(scale_factor=1)
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['gt_seg_map'].shape[:2], (32, 32))

    def test_repr(self):
        transform = SegRescale(scale_factor=2)
        self.assertEqual(
            repr(transform), ('SegRescale(scale_factor=2, backend=cv2)'))


class TestRandomCrop(unittest.TestCase):

    def test_init(self):
        # test invalid crop_type
        with self.assertRaisesRegex(ValueError, 'Invalid crop_type'):
            RandomCrop(crop_size=(10, 10), crop_type='unknown')

        crop_type_list = ['absolute', 'absolute_range']
        for crop_type in crop_type_list:
            # test h > 0 and w > 0
            for crop_size in [(0, 0), (0, 1), (1, 0)]:
                with self.assertRaises(AssertionError):
                    RandomCrop(crop_size=crop_size, crop_type=crop_type)
            # test type(h) = int and type(w) = int
            for crop_size in [(1.0, 1), (1, 1.0), (1.0, 1.0)]:
                with self.assertRaises(AssertionError):
                    RandomCrop(crop_size=crop_size, crop_type=crop_type)

        # test crop_size[0] <= crop_size[1]
        with self.assertRaises(AssertionError):
            RandomCrop(crop_size=(10, 5), crop_type='absolute_range')

        # test h in (0, 1] and w in (0, 1]
        crop_type_list = ['relative_range', 'relative']
        for crop_type in crop_type_list:
            for crop_size in [(0, 1), (1, 0), (1.1, 0.5), (0.5, 1.1)]:
                with self.assertRaises(AssertionError):
                    RandomCrop(crop_size=crop_size, crop_type=crop_type)

    def test_transform(self):
        # test relative and absolute crop
        src_results = {
            'img': np.random.randint(0, 255, size=(24, 32), dtype=np.int32)
        }
        target_shape = (12, 16)
        for crop_type, crop_size in zip(['relative', 'absolute'], [(0.5, 0.5),
                                                                   (16, 12)]):
            transform = RandomCrop(crop_size=crop_size, crop_type=crop_type)
            results = transform(copy.deepcopy(src_results))
            print(results['img'].shape[:2])
            self.assertEqual(results['img'].shape[:2], target_shape)

        # test absolute_range crop
        transform = RandomCrop(crop_size=(10, 20), crop_type='absolute_range')
        results = transform(copy.deepcopy(src_results))
        h, w = results['img'].shape
        self.assertTrue(10 <= w <= 20)
        self.assertTrue(10 <= h <= 20)
        self.assertEqual(results['img_shape'], results['img'].shape[:2])
        # test relative_range crop
        transform = RandomCrop(
            crop_size=(0.5, 0.5), crop_type='relative_range')
        results = transform(copy.deepcopy(src_results))
        h, w = results['img'].shape
        self.assertTrue(16 <= w <= 32)
        self.assertTrue(12 <= h <= 24)
        self.assertEqual(results['img_shape'], results['img'].shape[:2])

        # test with gt_bboxes, gt_bboxes_labels, gt_ignore_flags,
        # gt_masks, gt_seg_map, gt_instances_ids
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_bboxes = np.array([[0, 0, 7, 7], [2, 3, 9, 9]], dtype=np.float32)
        gt_bboxes_labels = np.array([0, 1], dtype=np.int64)
        gt_ignore_flags = np.array([0, 1], dtype=bool)
        gt_masks_ = np.zeros((2, 10, 10), np.uint8)
        gt_masks_[0, 0:7, 0:7] = 1
        gt_masks_[1, 2:7, 3:8] = 1
        gt_masks = BitmapMasks(gt_masks_.copy(), height=10, width=10)
        gt_seg_map = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_instances_ids = np.array([0, 1], dtype=np.int64)
        src_results = {
            'img': img,
            'gt_bboxes': gt_bboxes,
            'gt_bboxes_labels': gt_bboxes_labels,
            'gt_ignore_flags': gt_ignore_flags,
            'gt_masks': gt_masks,
            'gt_seg_map': gt_seg_map,
            'gt_instances_ids': gt_instances_ids
        }
        transform = RandomCrop(
            crop_size=(7, 5),
            allow_negative_crop=False,
            recompute_bbox=False,
            bbox_clip_border=True)
        results = transform(copy.deepcopy(src_results))
        h, w = results['img'].shape
        self.assertEqual(h, 5)
        self.assertEqual(w, 7)
        self.assertEqual(results['gt_bboxes'].shape[0], 2)
        self.assertEqual(results['gt_bboxes_labels'].shape[0], 2)
        self.assertEqual(results['gt_ignore_flags'].shape[0], 2)
        self.assertTupleEqual(results['gt_seg_map'].shape[:2], (5, 7))
        self.assertEqual(results['img_shape'], results['img'].shape[:2])
        self.assertEqual(results['gt_instances_ids'].shape[0], 2)

        # test geometric transformation with homography matrix
        bboxes = copy.deepcopy(src_results['gt_bboxes'])
        self.assertTrue((bbox_project(bboxes, results['homography_matrix'],
                                      (5, 7)) == results['gt_bboxes']).all())

        # test recompute_bbox = True
        gt_masks_ = np.zeros((2, 10, 10), np.uint8)
        gt_masks = BitmapMasks(gt_masks_.copy(), height=10, width=10)
        gt_bboxes = np.array([[0.1, 0.1, 0.2, 0.2]])
        src_results = {
            'img': img,
            'gt_bboxes': gt_bboxes,
            'gt_masks': gt_masks
        }
        target_gt_bboxes = np.zeros((1, 4), dtype=np.float32)
        transform = RandomCrop(
            crop_size=(10, 11),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=True)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue((results['gt_bboxes'] == target_gt_bboxes).all())

        # test bbox_clip_border = False
        src_results = {'img': img, 'gt_bboxes': gt_bboxes}
        transform = RandomCrop(
            crop_size=(10, 11),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=False)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue(
            (results['gt_bboxes'] == src_results['gt_bboxes']).all())

        # test the crop does not contain any gt-bbox
        # allow_negative_crop = False
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        src_results = {'img': img, 'gt_bboxes': gt_bboxes}
        transform = RandomCrop(crop_size=(5, 3), allow_negative_crop=False)
        results = transform(copy.deepcopy(src_results))
        self.assertIsNone(results)

        # allow_negative_crop = True
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        src_results = {'img': img, 'gt_bboxes': gt_bboxes}
        transform = RandomCrop(crop_size=(5, 3), allow_negative_crop=True)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue(isinstance(results, dict))

    def test_transform_use_box_type(self):
        # test with gt_bboxes, gt_bboxes_labels, gt_ignore_flags,
        # gt_masks, gt_seg_map
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_bboxes = np.array([[0, 0, 7, 7], [2, 3, 9, 9]], dtype=np.float32)
        gt_bboxes_labels = np.array([0, 1], dtype=np.int64)
        gt_ignore_flags = np.array([0, 1], dtype=bool)
        gt_masks_ = np.zeros((2, 10, 10), np.uint8)
        gt_masks_[0, 0:7, 0:7] = 1
        gt_masks_[1, 2:7, 3:8] = 1
        gt_masks = BitmapMasks(gt_masks_.copy(), height=10, width=10)
        gt_seg_map = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_instances_ids = np.array([0, 1], dtype=np.int64)
        src_results = {
            'img': img,
            'gt_bboxes': HorizontalBoxes(gt_bboxes),
            'gt_bboxes_labels': gt_bboxes_labels,
            'gt_ignore_flags': gt_ignore_flags,
            'gt_masks': gt_masks,
            'gt_seg_map': gt_seg_map,
            'gt_instances_ids': gt_instances_ids
        }
        transform = RandomCrop(
            crop_size=(7, 5),
            allow_negative_crop=False,
            recompute_bbox=False,
            bbox_clip_border=True)
        results = transform(copy.deepcopy(src_results))
        h, w = results['img'].shape
        self.assertEqual(h, 5)
        self.assertEqual(w, 7)
        self.assertEqual(results['gt_bboxes'].shape[0], 2)
        self.assertEqual(results['gt_bboxes_labels'].shape[0], 2)
        self.assertEqual(results['gt_ignore_flags'].shape[0], 2)
        self.assertTupleEqual(results['gt_seg_map'].shape[:2], (5, 7))
        self.assertEqual(results['gt_instances_ids'].shape[0], 2)

        # test geometric transformation with homography matrix
        bboxes = copy.deepcopy(src_results['gt_bboxes'].numpy())
        print(bboxes, results['gt_bboxes'])
        self.assertTrue(
            (bbox_project(bboxes, results['homography_matrix'],
                          (5, 7)) == results['gt_bboxes'].numpy()).all())

        # test recompute_bbox = True
        gt_masks_ = np.zeros((2, 10, 10), np.uint8)
        gt_masks = BitmapMasks(gt_masks_.copy(), height=10, width=10)
        gt_bboxes = HorizontalBoxes(np.array([[0.1, 0.1, 0.2, 0.2]]))
        src_results = {
            'img': img,
            'gt_bboxes': gt_bboxes,
            'gt_masks': gt_masks
        }
        target_gt_bboxes = np.zeros((1, 4), dtype=np.float32)
        transform = RandomCrop(
            crop_size=(10, 11),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=True)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue(
            (results['gt_bboxes'].numpy() == target_gt_bboxes).all())

        # test bbox_clip_border = False
        src_results = {'img': img, 'gt_bboxes': gt_bboxes}
        transform = RandomCrop(
            crop_size=(10, 10),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=False)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue(
            (results['gt_bboxes'].numpy() == src_results['gt_bboxes'].numpy()
             ).all())

        # test the crop does not contain any gt-bbox
        # allow_negative_crop = False
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_bboxes = HorizontalBoxes(np.zeros((0, 4), dtype=np.float32))
        src_results = {'img': img, 'gt_bboxes': gt_bboxes}
        transform = RandomCrop(crop_size=(5, 2), allow_negative_crop=False)
        results = transform(copy.deepcopy(src_results))
        self.assertIsNone(results)

        # allow_negative_crop = True
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_bboxes = HorizontalBoxes(np.zeros((0, 4), dtype=np.float32))
        src_results = {'img': img, 'gt_bboxes': gt_bboxes}
        transform = RandomCrop(crop_size=(5, 2), allow_negative_crop=True)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue(isinstance(results, dict))

    def test_repr(self):
        crop_type = 'absolute'
        crop_size = (10, 5)
        allow_negative_crop = False
        recompute_bbox = True
        bbox_clip_border = False
        transform = RandomCrop(
            crop_size=crop_size,
            crop_type=crop_type,
            allow_negative_crop=allow_negative_crop,
            recompute_bbox=recompute_bbox,
            bbox_clip_border=bbox_clip_border)
        self.assertEqual(
            repr(transform),
            f'RandomCrop(crop_size={crop_size}, crop_type={crop_type}, '
            f'allow_negative_crop={allow_negative_crop}, '
            f'recompute_bbox={recompute_bbox}, '
            f'bbox_clip_border={bbox_clip_border})')


class TestCutOut(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        self.results = {'img': img}

    def test_transform(self):
        # test n_holes
        with self.assertRaises(AssertionError):
            transform = CutOut(n_holes=(5, 3), cutout_shape=(8, 8))
        with self.assertRaises(AssertionError):
            transform = CutOut(n_holes=(3, 4, 5), cutout_shape=(8, 8))

        # test cutout_shape and cutout_ratio
        with self.assertRaises(AssertionError):
            transform = CutOut(n_holes=1, cutout_shape=8)
        with self.assertRaises(AssertionError):
            transform = CutOut(n_holes=1, cutout_ratio=0.2)

        # either of cutout_shape and cutout_ratio should be given
        with self.assertRaises(AssertionError):
            transform = CutOut(n_holes=1)
        with self.assertRaises(AssertionError):
            transform = CutOut(
                n_holes=1, cutout_shape=(2, 2), cutout_ratio=(0.4, 0.4))

        transform = CutOut(n_holes=1, cutout_shape=(10, 10))
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].sum() < self.results['img'].sum())

        transform = CutOut(
            n_holes=(2, 4),
            cutout_shape=[(10, 10), (15, 15)],
            fill_in=(255, 255, 255))
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].sum() > self.results['img'].sum())

        transform = CutOut(
            n_holes=1, cutout_ratio=(0.8, 0.8), fill_in=(255, 255, 255))
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].sum() > self.results['img'].sum())

    def test_repr(self):
        transform = CutOut(n_holes=1, cutout_shape=(10, 10))
        self.assertEqual(
            repr(transform), ('CutOut(n_holes=(1, 1), '
                              'cutout_shape=[(10, 10)], '
                              'fill_in=(0, 0, 0))'))
        transform = CutOut(
            n_holes=1, cutout_ratio=(0.8, 0.8), fill_in=(255, 255, 255))
        self.assertEqual(
            repr(transform), ('CutOut(n_holes=(1, 1), '
                              'cutout_ratio=[(0.8, 0.8)], '
                              'fill_in=(255, 255, 255))'))


class TestMosaic(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
            'gt_masks':
            BitmapMasks(rng.rand(3, 224, 224), height=224, width=224),
        }

    def test_transform(self):
        # test assertion for invalid img_scale
        with self.assertRaises(AssertionError):
            transform = Mosaic(img_scale=640)

        # test assertion for invalid probability
        with self.assertRaises(AssertionError):
            transform = Mosaic(prob=1.5)

        transform = Mosaic(img_scale=(12, 10))
        # test assertion for invalid mix_results
        with self.assertRaises(AssertionError):
            results = transform(copy.deepcopy(self.results))

        self.results['mix_results'] = [copy.deepcopy(self.results)] * 3
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)
        self.assertEqual(results['img_shape'], results['img'].shape[:2])

    def test_transform_with_no_gt(self):
        self.results['gt_bboxes'] = np.empty((0, 4), dtype=np.float32)
        self.results['gt_bboxes_labels'] = np.empty((0, ), dtype=np.int64)
        self.results['gt_ignore_flags'] = np.empty((0, ), dtype=bool)
        transform = Mosaic(img_scale=(12, 10))
        self.results['mix_results'] = [copy.deepcopy(self.results)] * 3
        results = transform(copy.deepcopy(self.results))
        self.assertIsInstance(results, dict)
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(
            results['gt_bboxes_labels'].shape[0] == results['gt_bboxes'].
            shape[0] == results['gt_ignore_flags'].shape[0] == 0)
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_use_box_type(self):
        transform = Mosaic(img_scale=(12, 10))
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])
        results['mix_results'] = [results] * 3
        results = transform(results)
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_repr(self):
        transform = Mosaic(img_scale=(640, 640), )
        self.assertEqual(
            repr(transform), ('Mosaic(img_scale=(640, 640), '
                              'center_ratio_range=(0.5, 1.5), '
                              'pad_val=114.0, '
                              'prob=1.0)'))


class TestMixUp(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
            'gt_masks':
            BitmapMasks(rng.rand(3, 224, 224), height=224, width=224),
        }

    def test_transform(self):
        # test assertion for invalid img_scale
        with self.assertRaises(AssertionError):
            transform = MixUp(img_scale=640)

        transform = MixUp(img_scale=(12, 10))
        # test assertion for invalid mix_results
        with self.assertRaises(AssertionError):
            results = transform(copy.deepcopy(self.results))

        with self.assertRaises(AssertionError):
            self.results['mix_results'] = [copy.deepcopy(self.results)] * 2
            results = transform(copy.deepcopy(self.results))

        self.results['mix_results'] = [copy.deepcopy(self.results)]
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)
        self.assertEqual(results['img_shape'], results['img'].shape[:2])

    def test_transform_use_box_type(self):
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])

        transform = MixUp(img_scale=(12, 10))
        results['mix_results'] = [results]
        results = transform(results)
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_repr(self):
        transform = MixUp(
            img_scale=(640, 640),
            ratio_range=(0.8, 1.6),
            pad_val=114.0,
        )
        self.assertEqual(
            repr(transform), ('MixUp(dynamic_scale=(640, 640), '
                              'ratio_range=(0.8, 1.6), '
                              'flip_ratio=0.5, '
                              'pad_val=114.0, '
                              'max_iters=15, '
                              'bbox_clip_border=True)'))


class TestRandomAffine(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
        }

    def test_transform(self):
        # test assertion for invalid translate_ratio
        with self.assertRaises(AssertionError):
            transform = RandomAffine(max_translate_ratio=1.5)

        # test assertion for invalid scaling_ratio_range
        with self.assertRaises(AssertionError):
            transform = RandomAffine(scaling_ratio_range=(1.5, 0.5))

        with self.assertRaises(AssertionError):
            transform = RandomAffine(scaling_ratio_range=(0, 0.5))

        transform = RandomAffine()
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)
        self.assertEqual(results['img_shape'], results['img'].shape[:2])

    def test_transform_use_box_type(self):
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])

        transform = RandomAffine()
        results = transform(copy.deepcopy(results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_repr(self):
        transform = RandomAffine(
            scaling_ratio_range=(0.1, 2),
            border=(-320, -320),
        )
        self.assertEqual(
            repr(transform), ('RandomAffine(max_rotate_degree=10.0, '
                              'max_translate_ratio=0.1, '
                              'scaling_ratio_range=(0.1, 2), '
                              'max_shear_degree=2.0, '
                              'border=(-320, -320), '
                              'border_val=(114, 114, 114), '
                              'bbox_clip_border=True)'))


class TestYOLOXHSVRandomAug(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        self.results = {
            'img':
            img,
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
        }

    def test_transform(self):
        transform = YOLOXHSVRandomAug()
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(
            results['img'].shape[:2] == self.results['img'].shape[:2])
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_repr(self):
        transform = YOLOXHSVRandomAug()
        self.assertEqual(
            repr(transform), ('YOLOXHSVRandomAug(hue_delta=5, '
                              'saturation_delta=30, '
                              'value_delta=30)'))


class TestRandomCenterCropPad(unittest.TestCase):

    def test_init(self):
        # test assertion for invalid crop_size while test_mode=False
        with self.assertRaises(AssertionError):
            RandomCenterCropPad(
                crop_size=(-1, 0), test_mode=False, test_pad_mode=None)

        # test assertion for invalid ratios while test_mode=False
        with self.assertRaises(AssertionError):
            RandomCenterCropPad(
                crop_size=(511, 511),
                ratios=(1.0, 1.0),
                test_mode=False,
                test_pad_mode=None)

        # test assertion for invalid mean, std and to_rgb
        with self.assertRaises(AssertionError):
            RandomCenterCropPad(
                crop_size=(511, 511),
                mean=None,
                std=None,
                to_rgb=None,
                test_mode=False,
                test_pad_mode=None)

        # test assertion for invalid crop_size while test_mode=True
        with self.assertRaises(AssertionError):
            RandomCenterCropPad(
                crop_size=(511, 511),
                ratios=None,
                border=None,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=('logical_or', 127))

        # test assertion for invalid ratios while test_mode=True
        with self.assertRaises(AssertionError):
            RandomCenterCropPad(
                crop_size=None,
                ratios=(0.9, 1.0, 1.1),
                border=None,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=('logical_or', 127))

        # test assertion for invalid border while test_mode=True
        with self.assertRaises(AssertionError):
            RandomCenterCropPad(
                crop_size=None,
                ratios=None,
                border=128,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=('logical_or', 127))

        # test assertion for invalid test_pad_mode while test_mode=True
        with self.assertRaises(AssertionError):
            RandomCenterCropPad(
                crop_size=None,
                ratios=None,
                border=None,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=('do_nothing', 100))

    def test_transform(self):
        results = dict(
            img_path=osp.join(osp.dirname(__file__), '../../data/color.jpg'))

        load = LoadImageFromFile(to_float32=True)
        results = load(results)
        test_results = copy.deepcopy(results)

        h, w = results['img_shape']
        gt_bboxes = create_random_bboxes(4, w, h)
        gt_bboxes_labels = np.array([1, 2, 3, 1], dtype=np.int64)
        gt_ignore_flags = np.array([0, 0, 1, 1], dtype=bool)
        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_bboxes_labels
        results['gt_ignore_flags'] = gt_ignore_flags
        crop_module = RandomCenterCropPad(
            crop_size=(w - 20, h - 20),
            ratios=(1.0, ),
            border=128,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=False,
            test_pad_mode=None)
        train_results = crop_module(results)
        assert train_results['img'].shape[:2] == (h - 20, w - 20)
        # All bboxes should be reserved after crop
        assert train_results['img_shape'][:2] == (h - 20, w - 20)
        assert train_results['gt_bboxes'].shape[0] == 4
        assert train_results['gt_bboxes'].dtype == np.float32
        self.assertEqual(results['img_shape'], results['img'].shape[:2])

        crop_module = RandomCenterCropPad(
            crop_size=None,
            ratios=None,
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        test_results = crop_module(test_results)
        assert test_results['img'].shape[:2] == (h | 127, w | 127)
        assert test_results['img_shape'][:2] == (h | 127, w | 127)
        assert 'border' in test_results

    def test_transform_use_box_type(self):
        results = dict(
            img_path=osp.join(osp.dirname(__file__), '../../data/color.jpg'))

        load = LoadImageFromFile(to_float32=True)
        results = load(results)
        test_results = copy.deepcopy(results)

        h, w = results['img_shape']
        gt_bboxes = create_random_bboxes(4, w, h)
        gt_bboxes_labels = np.array([1, 2, 3, 1], dtype=np.int64)
        gt_ignore_flags = np.array([0, 0, 1, 1], dtype=bool)
        results['gt_bboxes'] = HorizontalBoxes(gt_bboxes)
        results['gt_bboxes_labels'] = gt_bboxes_labels
        results['gt_ignore_flags'] = gt_ignore_flags
        crop_module = RandomCenterCropPad(
            crop_size=(w - 20, h - 20),
            ratios=(1.0, ),
            border=128,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=False,
            test_pad_mode=None)
        train_results = crop_module(results)
        assert train_results['img'].shape[:2] == (h - 20, w - 20)
        # All bboxes should be reserved after crop
        assert train_results['img_shape'][:2] == (h - 20, w - 20)
        assert train_results['gt_bboxes'].shape[0] == 4
        assert train_results['gt_bboxes'].dtype == torch.float32

        crop_module = RandomCenterCropPad(
            crop_size=None,
            ratios=None,
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        test_results = crop_module(test_results)
        assert test_results['img'].shape[:2] == (h | 127, w | 127)
        assert test_results['img_shape'][:2] == (h | 127, w | 127)
        assert 'border' in test_results


class TestCopyPaste(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        h, w, _ = img.shape
        dst_bboxes = np.array([[0.2 * w, 0.2 * h, 0.4 * w, 0.4 * h],
                               [0.5 * w, 0.5 * h, 0.6 * w, 0.6 * h]],
                              dtype=np.float32)
        src_bboxes = np.array([[0.1 * w, 0.1 * h, 0.3 * w, 0.5 * h],
                               [0.4 * w, 0.4 * h, 0.7 * w, 0.7 * h],
                               [0.8 * w, 0.8 * h, 0.9 * w, 0.9 * h]],
                              dtype=np.float32)

        self.dst_results = {
            'img': img.copy(),
            'gt_bboxes': dst_bboxes,
            'gt_bboxes_labels': np.ones(dst_bboxes.shape[0], dtype=np.int64),
            'gt_masks': create_full_masks(dst_bboxes, w, h),
            'gt_ignore_flags': np.array([0, 1], dtype=bool),
        }
        self.src_results = {
            'img': img.copy(),
            'gt_bboxes': src_bboxes,
            'gt_bboxes_labels':
            np.ones(src_bboxes.shape[0], dtype=np.int64) * 2,
            'gt_masks': create_full_masks(src_bboxes, w, h),
            'gt_ignore_flags': np.array([0, 0, 1], dtype=bool),
        }

    def test_transform(self):
        transform = CopyPaste(selected=False)
        # test assertion for invalid mix_results
        with self.assertRaises(AssertionError):
            results = transform(copy.deepcopy(self.dst_results))

        results = copy.deepcopy(self.dst_results)
        results['mix_results'] = [copy.deepcopy(self.src_results)]
        results = transform(results)

        self.assertEqual(results['img'].shape[:2],
                         self.dst_results['img'].shape[:2])

        # one object of destination image is totally occluded
        self.assertEqual(
            results['gt_bboxes'].shape[0],
            self.dst_results['gt_bboxes'].shape[0] +
            self.src_results['gt_bboxes'].shape[0] - 1)
        self.assertEqual(
            results['gt_bboxes_labels'].shape[0],
            self.dst_results['gt_bboxes_labels'].shape[0] +
            self.src_results['gt_bboxes_labels'].shape[0] - 1)
        self.assertEqual(
            results['gt_masks'].masks.shape[0],
            self.dst_results['gt_masks'].masks.shape[0] +
            self.src_results['gt_masks'].masks.shape[0] - 1)
        self.assertEqual(
            results['gt_ignore_flags'].shape[0],
            self.dst_results['gt_ignore_flags'].shape[0] +
            self.src_results['gt_ignore_flags'].shape[0] - 1)

        # the object of destination image is partially occluded
        ori_bbox = self.dst_results['gt_bboxes'][0]
        occ_bbox = results['gt_bboxes'][0]
        ori_mask = self.dst_results['gt_masks'].masks[0]
        occ_mask = results['gt_masks'].masks[0]
        self.assertTrue(ori_mask.sum() > occ_mask.sum())
        self.assertTrue(
            np.all(np.abs(occ_bbox - ori_bbox) <= transform.bbox_occluded_thr)
            or occ_mask.sum() > transform.mask_occluded_thr)

        # test copypaste with selected objects
        transform = CopyPaste()
        results = copy.deepcopy(self.dst_results)
        results['mix_results'] = [copy.deepcopy(self.src_results)]
        results = transform(results)

        # test copypaste with an empty source image
        results = copy.deepcopy(self.dst_results)
        valid_inds = [False] * self.src_results['gt_bboxes'].shape[0]
        results['mix_results'] = [{
            'img':
            self.src_results['img'].copy(),
            'gt_bboxes':
            self.src_results['gt_bboxes'][valid_inds],
            'gt_bboxes_labels':
            self.src_results['gt_bboxes_labels'][valid_inds],
            'gt_masks':
            self.src_results['gt_masks'][valid_inds],
            'gt_ignore_flags':
            self.src_results['gt_ignore_flags'][valid_inds],
        }]
        results = transform(results)

    def test_transform_use_box_type(self):
        src_results = copy.deepcopy(self.src_results)
        src_results['gt_bboxes'] = HorizontalBoxes(src_results['gt_bboxes'])
        dst_results = copy.deepcopy(self.dst_results)
        dst_results['gt_bboxes'] = HorizontalBoxes(dst_results['gt_bboxes'])
        transform = CopyPaste(selected=False)

        results = copy.deepcopy(dst_results)
        results['mix_results'] = [copy.deepcopy(src_results)]
        results = transform(results)

        self.assertEqual(results['img'].shape[:2],
                         self.dst_results['img'].shape[:2])

        # one object of destination image is totally occluded
        self.assertEqual(
            results['gt_bboxes'].shape[0],
            self.dst_results['gt_bboxes'].shape[0] +
            self.src_results['gt_bboxes'].shape[0] - 1)
        self.assertEqual(
            results['gt_bboxes_labels'].shape[0],
            self.dst_results['gt_bboxes_labels'].shape[0] +
            self.src_results['gt_bboxes_labels'].shape[0] - 1)
        self.assertEqual(
            results['gt_masks'].masks.shape[0],
            self.dst_results['gt_masks'].masks.shape[0] +
            self.src_results['gt_masks'].masks.shape[0] - 1)
        self.assertEqual(
            results['gt_ignore_flags'].shape[0],
            self.dst_results['gt_ignore_flags'].shape[0] +
            self.src_results['gt_ignore_flags'].shape[0] - 1)

        # the object of destination image is partially occluded
        ori_bbox = dst_results['gt_bboxes'][0].numpy()
        occ_bbox = results['gt_bboxes'][0].numpy()
        ori_mask = dst_results['gt_masks'].masks[0]
        occ_mask = results['gt_masks'].masks[0]
        self.assertTrue(ori_mask.sum() > occ_mask.sum())
        self.assertTrue(
            np.all(np.abs(occ_bbox - ori_bbox) <= transform.bbox_occluded_thr)
            or occ_mask.sum() > transform.mask_occluded_thr)

        # test copypaste with selected objects
        transform = CopyPaste()
        results = copy.deepcopy(dst_results)
        results['mix_results'] = [copy.deepcopy(src_results)]
        results = transform(results)

        # test copypaste with an empty source image
        results = copy.deepcopy(dst_results)
        valid_inds = [False] * self.src_results['gt_bboxes'].shape[0]
        results['mix_results'] = [{
            'img':
            src_results['img'].copy(),
            'gt_bboxes':
            src_results['gt_bboxes'][valid_inds],
            'gt_bboxes_labels':
            src_results['gt_bboxes_labels'][valid_inds],
            'gt_masks':
            src_results['gt_masks'][valid_inds],
            'gt_ignore_flags':
            src_results['gt_ignore_flags'][valid_inds],
        }]
        results = transform(results)

    def test_repr(self):
        transform = CopyPaste()
        self.assertEqual(
            repr(transform), ('CopyPaste(max_num_pasted=100, '
                              'bbox_occluded_thr=10, '
                              'mask_occluded_thr=300, '
                              'selected=True)'))


class TestAlbu(unittest.TestCase):

    @unittest.skipIf(albumentations is None, 'albumentations is not installed')
    def test_transform(self):
        results = dict(
            img_path=osp.join(osp.dirname(__file__), '../../data/color.jpg'))

        # Define simple pipeline
        load = dict(type='LoadImageFromFile')
        load = TRANSFORMS.build(load)

        albu_transform = dict(
            type='Albu', transforms=[dict(type='ChannelShuffle', p=1)])
        albu_transform = TRANSFORMS.build(albu_transform)

        # Execute transforms
        results = load(results)
        results = albu_transform(results)

        self.assertEqual(results['img'].dtype, np.uint8)

        # test bbox
        albu_transform = dict(
            type='Albu',
            transforms=[dict(type='ChannelShuffle', p=1)],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
            keymap={
                'img': 'image',
                'gt_bboxes': 'bboxes'
            })
        albu_transform = TRANSFORMS.build(albu_transform)
        results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
        }
        results = albu_transform(results)
        self.assertEqual(results['img'].dtype, np.float64)
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)
        self.assertEqual(results['gt_ignore_flags'].dtype, bool)
        self.assertEqual(results['gt_bboxes_labels'].dtype, np.int64)
        self.assertEqual(results['img_shape'], results['img'].shape[:2])

    @unittest.skipIf(albumentations is None, 'albumentations is not installed')
    def test_repr(self):
        albu_transform = dict(
            type='Albu', transforms=[dict(type='ChannelShuffle', p=1)])
        albu_transform = TRANSFORMS.build(albu_transform)

        self.assertEqual(
            repr(albu_transform), 'Albu(transforms=['
            '{\'type\': \'ChannelShuffle\', '
            '\'p\': 1}])')


class TestCorrupt(unittest.TestCase):

    def test_transform(self):
        results = dict(
            img_path=osp.join(osp.dirname(__file__), '../../data/color.jpg'))

        # Define simple pipeline
        load = dict(type='LoadImageFromFile')
        load = TRANSFORMS.build(load)

        corrupt_transform = dict(type='Corrupt', corruption='gaussian_blur')
        corrupt_transform = TRANSFORMS.build(corrupt_transform)

        # Execute transforms
        results = load(results)
        results = corrupt_transform(results)

        self.assertEqual(results['img'].dtype, np.uint8)

    def test_repr(self):
        corrupt_transform = dict(type='Corrupt', corruption='gaussian_blur')
        corrupt_transform = TRANSFORMS.build(corrupt_transform)

        self.assertEqual(
            repr(corrupt_transform), 'Corrupt(corruption=gaussian_blur, '
            'severity=1)')


class TestRandomShift(unittest.TestCase):

    def test_init(self):
        # test assertion for invalid shift_ratio
        with self.assertRaises(AssertionError):
            RandomShift(prob=1.5)

        # test assertion for invalid max_shift_px
        with self.assertRaises(AssertionError):
            RandomShift(max_shift_px=-1)

    def test_transform(self):

        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        results['img'] = img
        h, w, _ = img.shape
        gt_bboxes = create_random_bboxes(8, w, h)
        results['gt_bboxes_labels'] = np.ones(
            gt_bboxes.shape[0], dtype=np.int64)
        results['gt_bboxes'] = gt_bboxes
        transform = RandomShift(prob=1.0)
        results = transform(results)

        self.assertEqual(results['img'].shape[:2], (h, w))
        self.assertEqual(results['gt_bboxes_labels'].shape[0],
                         results['gt_bboxes'].shape[0])
        self.assertEqual(results['gt_bboxes_labels'].dtype, np.int64)
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)

    def test_transform_use_box_type(self):

        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        results['img'] = img
        h, w, _ = img.shape
        gt_bboxes = create_random_bboxes(8, w, h)
        results['gt_bboxes_labels'] = np.ones(
            gt_bboxes.shape[0], dtype=np.int64)
        results['gt_bboxes'] = HorizontalBoxes(gt_bboxes)
        transform = RandomShift(prob=1.0)
        results = transform(results)

        self.assertEqual(results['img'].shape[:2], (h, w))
        self.assertEqual(results['gt_bboxes_labels'].shape[0],
                         results['gt_bboxes'].shape[0])
        self.assertEqual(results['gt_bboxes_labels'].dtype, np.int64)
        self.assertEqual(results['gt_bboxes'].dtype, torch.float32)

    def test_repr(self):
        transform = RandomShift()
        self.assertEqual(
            repr(transform), ('RandomShift(prob=0.5, '
                              'max_shift_px=32, '
                              'filter_thr_px=1)'))


class TestRandomErasing(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.results = construct_toy_data(poly2mask=True)

    def test_transform(self):
        transform = RandomErasing(
            n_patches=(1, 5), ratio=(0.4, 0.8), img_border_value=0)
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].sum() < self.results['img'].sum())

        transform = RandomErasing(
            n_patches=1, ratio=0.999, img_border_value=255)
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].sum() > self.results['img'].sum())
        # test empty results
        empty_results = copy.deepcopy(self.results)
        empty_results['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
        empty_results['gt_bboxes_labels'] = np.zeros((0, ), dtype=np.int64)
        empty_results['gt_masks'] = empty_results['gt_masks'][False]
        empty_results['gt_ignore_flags'] = np.zeros((0, ), dtype=bool)
        empty_results['gt_seg_map'] = np.ones_like(
            empty_results['gt_seg_map']) * 255
        results = transform(copy.deepcopy(empty_results))
        self.assertTrue(results['img'].sum() > self.results['img'].sum())

    def test_transform_use_box_type(self):
        src_results = copy.deepcopy(self.results)
        src_results['gt_bboxes'] = HorizontalBoxes(src_results['gt_bboxes'])

        transform = RandomErasing(
            n_patches=(1, 5), ratio=(0.4, 0.8), img_border_value=0)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue(results['img'].sum() < src_results['img'].sum())

        transform = RandomErasing(
            n_patches=1, ratio=0.999, img_border_value=255)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue(results['img'].sum() > src_results['img'].sum())
        # test empty results
        empty_results = copy.deepcopy(src_results)
        empty_results['gt_bboxes'] = HorizontalBoxes([], dtype=torch.float32)
        empty_results['gt_bboxes_labels'] = np.zeros((0, ), dtype=np.int64)
        empty_results['gt_masks'] = empty_results['gt_masks'][False]
        empty_results['gt_ignore_flags'] = np.zeros((0, ), dtype=bool)
        empty_results['gt_seg_map'] = np.ones_like(
            empty_results['gt_seg_map']) * 255
        results = transform(copy.deepcopy(empty_results))
        self.assertTrue(results['img'].sum() > src_results['img'].sum())

    def test_repr(self):
        transform = RandomErasing(n_patches=(1, 5), ratio=(0, 0.2))
        self.assertEqual(
            repr(transform), ('RandomErasing(n_patches=(1, 5), '
                              'ratio=(0, 0.2), '
                              'squared=True, '
                              'bbox_erased_thr=0.9, '
                              'img_border_value=128, '
                              'mask_border_value=0, '
                              'seg_ignore_label=255)'))
