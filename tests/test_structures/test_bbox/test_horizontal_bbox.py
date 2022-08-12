import random
from math import sqrt
from unittest import TestCase

import cv2
import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks, PolygonMasks


class TestHorizontalBoxes(TestCase):

    def test_init(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        th_bboxes_cxcywh = torch.Tensor([15, 15, 10, 10]).reshape(1, 1, 4)

        bboxes = HorizontalBoxes(th_bboxes)
        assert_allclose(bboxes.tensor, th_bboxes)
        bboxes = HorizontalBoxes(th_bboxes, pattern='xyxy')
        assert_allclose(bboxes.tensor, th_bboxes)
        bboxes = HorizontalBoxes(th_bboxes_cxcywh, pattern='cxcywh')
        assert_allclose(bboxes.tensor, th_bboxes)
        with self.assertRaises(ValueError):
            bboxes = HorizontalBoxes(th_bboxes, pattern='invalid')

    def test_cxcywh(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        th_bboxes_cxcywh = torch.Tensor([15, 15, 10, 10]).reshape(1, 1, 4)
        bboxes = HorizontalBoxes(th_bboxes)

        assert_allclose(
            HorizontalBoxes.xyxy_to_cxcywh(th_bboxes), th_bboxes_cxcywh)
        assert_allclose(th_bboxes,
                        HorizontalBoxes.cxcywh_to_xyxy(th_bboxes_cxcywh))
        assert_allclose(bboxes.cxcywh, th_bboxes_cxcywh)

    def test_propoerty(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        bboxes = HorizontalBoxes(th_bboxes)

        # Centers
        centers = torch.Tensor([15, 15]).reshape(1, 1, 2)
        assert_allclose(bboxes.centers, centers)
        # Areas
        areas = torch.Tensor([100]).reshape(1, 1)
        assert_allclose(bboxes.areas, areas)
        # widths
        widths = torch.Tensor([10]).reshape(1, 1)
        assert_allclose(bboxes.widths, widths)
        # heights
        heights = torch.Tensor([10]).reshape(1, 1)
        assert_allclose(bboxes.heights, heights)

    def test_flip(self):
        img_shape = [50, 85]
        # horizontal flip
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        flipped_bboxes_th = torch.Tensor([65, 10, 75, 20]).reshape(1, 1, 4)
        bboxes = HorizontalBoxes(th_bboxes)
        bboxes.flip_(img_shape, direction='horizontal')
        assert_allclose(bboxes.tensor, flipped_bboxes_th)
        # vertical flip
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        flipped_bboxes_th = torch.Tensor([10, 30, 20, 40]).reshape(1, 1, 4)
        bboxes = HorizontalBoxes(th_bboxes)
        bboxes.flip_(img_shape, direction='vertical')
        assert_allclose(bboxes.tensor, flipped_bboxes_th)
        # diagonal flip
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        flipped_bboxes_th = torch.Tensor([65, 30, 75, 40]).reshape(1, 1, 4)
        bboxes = HorizontalBoxes(th_bboxes)
        bboxes.flip_(img_shape, direction='diagonal')
        assert_allclose(bboxes.tensor, flipped_bboxes_th)

    def test_translate(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        bboxes = HorizontalBoxes(th_bboxes)
        bboxes.translate_([23, 46])
        translated_bboxes_th = torch.Tensor([33, 56, 43, 66]).reshape(1, 1, 4)
        assert_allclose(bboxes.tensor, translated_bboxes_th)

    def test_clip(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        img_shape = [13, 14]
        bboxes = HorizontalBoxes(th_bboxes)
        bboxes.clip_(img_shape)
        cliped_bboxes_th = torch.Tensor([10, 10, 14, 13]).reshape(1, 1, 4)
        assert_allclose(bboxes.tensor, cliped_bboxes_th)

    def test_rotate(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        center = (15, 15)
        angle = -45
        bboxes = HorizontalBoxes(th_bboxes)
        bboxes.rotate_(center, angle)
        rotated_bboxes_th = torch.Tensor([
            15 - 5 * sqrt(2), 15 - 5 * sqrt(2), 15 + 5 * sqrt(2),
            15 + 5 * sqrt(2)
        ]).reshape(1, 1, 4)
        assert_allclose(bboxes.tensor, rotated_bboxes_th)

    def test_project(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        bboxes1 = HorizontalBoxes(th_bboxes)
        bboxes2 = bboxes1.clone()

        matrix = np.zeros((3, 3), dtype=np.float32)
        center = [random.random() * 80, random.random() * 80]
        angle = random.random() * 180
        matrix[:2, :3] = cv2.getRotationMatrix2D(center, angle, 1)
        x_translate = random.random() * 40
        y_translate = random.random() * 40
        matrix[0, 2] = matrix[0, 2] + x_translate
        matrix[1, 2] = matrix[1, 2] + y_translate
        scale_factor = random.random() * 2
        matrix[2, 2] = 1 / scale_factor
        bboxes1.project_(matrix)

        bboxes2.rotate_(center, -angle)
        bboxes2.translate_([x_translate, y_translate])
        bboxes2.rescale_([scale_factor, scale_factor])
        assert_allclose(bboxes1.tensor, bboxes2.tensor)

    def test_rescale(self):
        scale_factor = [0.4, 0.8]
        # rescale
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        bboxes = HorizontalBoxes(th_bboxes)
        bboxes.rescale_(scale_factor)
        rescaled_bboxes_th = torch.Tensor([4, 8, 8, 16]).reshape(1, 1, 4)
        assert_allclose(bboxes.tensor, rescaled_bboxes_th)

    def test_resize(self):
        scale_factor = [0.4, 0.8]
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        bboxes = HorizontalBoxes(th_bboxes)
        bboxes.resize_(scale_factor)
        resized_bboxes_th = torch.Tensor([13, 11, 17, 19]).reshape(1, 1, 4)
        assert_allclose(bboxes.tensor, resized_bboxes_th)

    def test_is_bboxes_inside(self):
        th_bboxes = torch.Tensor([[10, 10, 20, 20], [-5, -5, 15, 15],
                                  [45, 45, 55, 55]]).reshape(1, 3, 4)
        img_shape = [30, 30]
        bboxes = HorizontalBoxes(th_bboxes)

        index = bboxes.is_bboxes_inside(img_shape)
        index_th = torch.BoolTensor([True, True, False]).reshape(1, 3)
        assert_allclose(index, index_th)

    def test_find_inside_points(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 4)
        bboxes = HorizontalBoxes(th_bboxes)
        points = torch.Tensor([[0, 0], [0, 15], [15, 0], [15, 15]])
        index = bboxes.find_inside_points(points)
        index_th = torch.BoolTensor([False, False, False, True]).reshape(4, 1)
        assert_allclose(index, index_th)
        # is_aligned
        bboxes = bboxes.expand(4, 4)
        index = bboxes.find_inside_points(points, is_aligned=True)
        index_th = torch.BoolTensor([False, False, False, True])
        assert_allclose(index, index_th)

    def test_from_masks(self):
        bitmap_masks = BitmapMasks.random()
        bboxes = HorizontalBoxes.from_bitmap_masks(bitmap_masks)
        self.assertIsInstance(bboxes, HorizontalBoxes)
        self.assertEqual(len(bboxes), len(bitmap_masks))
        polygon_masks = PolygonMasks.random()
        bboxes = HorizontalBoxes.from_polygon_masks(polygon_masks)
        self.assertIsInstance(bboxes, HorizontalBoxes)
        self.assertEqual(len(bboxes), len(bitmap_masks))
        # zero length masks
        bitmap_masks = BitmapMasks.random(num_masks=0)
        bboxes = HorizontalBoxes.from_bitmap_masks(bitmap_masks)
        self.assertIsInstance(bboxes, HorizontalBoxes)
        self.assertEqual(len(bboxes), 0)
        polygon_masks = PolygonMasks.random(num_masks=0)
        bboxes = HorizontalBoxes.from_polygon_masks(polygon_masks)
        self.assertIsInstance(bboxes, HorizontalBoxes)
        self.assertEqual(len(bboxes), 0)
