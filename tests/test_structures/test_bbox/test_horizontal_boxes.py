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
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        th_boxes_cxcywh = torch.Tensor([15, 15, 10, 10]).reshape(1, 1, 4)

        boxes = HorizontalBoxes(th_boxes)
        assert_allclose(boxes.tensor, th_boxes)
        boxes = HorizontalBoxes(th_boxes, in_mode='xyxy')
        assert_allclose(boxes.tensor, th_boxes)
        boxes = HorizontalBoxes(th_boxes_cxcywh, in_mode='cxcywh')
        assert_allclose(boxes.tensor, th_boxes)
        with self.assertRaises(ValueError):
            boxes = HorizontalBoxes(th_boxes, in_mode='invalid')

    def test_cxcywh(self):
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        th_boxes_cxcywh = torch.Tensor([15, 15, 10, 10]).reshape(1, 1, 4)
        boxes = HorizontalBoxes(th_boxes)

        assert_allclose(
            HorizontalBoxes.xyxy_to_cxcywh(th_boxes), th_boxes_cxcywh)
        assert_allclose(th_boxes,
                        HorizontalBoxes.cxcywh_to_xyxy(th_boxes_cxcywh))
        assert_allclose(boxes.cxcywh, th_boxes_cxcywh)

    def test_propoerty(self):
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        boxes = HorizontalBoxes(th_boxes)

        # Centers
        centers = torch.Tensor([15, 15]).reshape(1, 1, 2)
        assert_allclose(boxes.centers, centers)
        # Areas
        areas = torch.Tensor([100]).reshape(1, 1)
        assert_allclose(boxes.areas, areas)
        # widths
        widths = torch.Tensor([10]).reshape(1, 1)
        assert_allclose(boxes.widths, widths)
        # heights
        heights = torch.Tensor([10]).reshape(1, 1)
        assert_allclose(boxes.heights, heights)

    def test_flip(self):
        img_shape = [50, 85]
        # horizontal flip
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        flipped_boxes_th = torch.Tensor([65, 10, 75, 20]).reshape(1, 1, 4)
        boxes = HorizontalBoxes(th_boxes)
        boxes.flip_(img_shape, direction='horizontal')
        assert_allclose(boxes.tensor, flipped_boxes_th)
        # vertical flip
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        flipped_boxes_th = torch.Tensor([10, 30, 20, 40]).reshape(1, 1, 4)
        boxes = HorizontalBoxes(th_boxes)
        boxes.flip_(img_shape, direction='vertical')
        assert_allclose(boxes.tensor, flipped_boxes_th)
        # diagonal flip
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        flipped_boxes_th = torch.Tensor([65, 30, 75, 40]).reshape(1, 1, 4)
        boxes = HorizontalBoxes(th_boxes)
        boxes.flip_(img_shape, direction='diagonal')
        assert_allclose(boxes.tensor, flipped_boxes_th)

    def test_translate(self):
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        boxes = HorizontalBoxes(th_boxes)
        boxes.translate_([23, 46])
        translated_boxes_th = torch.Tensor([33, 56, 43, 66]).reshape(1, 1, 4)
        assert_allclose(boxes.tensor, translated_boxes_th)

    def test_clip(self):
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        img_shape = [13, 14]
        boxes = HorizontalBoxes(th_boxes)
        boxes.clip_(img_shape)
        cliped_boxes_th = torch.Tensor([10, 10, 14, 13]).reshape(1, 1, 4)
        assert_allclose(boxes.tensor, cliped_boxes_th)

    def test_rotate(self):
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        center = (15, 15)
        angle = -45
        boxes = HorizontalBoxes(th_boxes)
        boxes.rotate_(center, angle)
        rotated_boxes_th = torch.Tensor([
            15 - 5 * sqrt(2), 15 - 5 * sqrt(2), 15 + 5 * sqrt(2),
            15 + 5 * sqrt(2)
        ]).reshape(1, 1, 4)
        assert_allclose(boxes.tensor, rotated_boxes_th)

    def test_project(self):
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        boxes1 = HorizontalBoxes(th_boxes)
        boxes2 = boxes1.clone()

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
        boxes1.project_(matrix)

        boxes2.rotate_(center, -angle)
        boxes2.translate_([x_translate, y_translate])
        boxes2.rescale_([scale_factor, scale_factor])
        assert_allclose(boxes1.tensor, boxes2.tensor)
        # test empty boxes
        empty_boxes = HorizontalBoxes(torch.zeros((0, 4)))
        empty_boxes.project_(matrix)

    def test_rescale(self):
        scale_factor = [0.4, 0.8]
        # rescale
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        boxes = HorizontalBoxes(th_boxes)
        boxes.rescale_(scale_factor)
        rescaled_boxes_th = torch.Tensor([4, 8, 8, 16]).reshape(1, 1, 4)
        assert_allclose(boxes.tensor, rescaled_boxes_th)

    def test_resize(self):
        scale_factor = [0.4, 0.8]
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        boxes = HorizontalBoxes(th_boxes)
        boxes.resize_(scale_factor)
        resized_boxes_th = torch.Tensor([13, 11, 17, 19]).reshape(1, 1, 4)
        assert_allclose(boxes.tensor, resized_boxes_th)

    def test_is_inside(self):
        th_boxes = torch.Tensor([[10, 10, 20, 20], [-5, -5, 15, 15],
                                 [45, 45, 55, 55]]).reshape(1, 3, 4)
        img_shape = [30, 30]
        boxes = HorizontalBoxes(th_boxes)

        index = boxes.is_inside(img_shape)
        index_th = torch.BoolTensor([True, True, False]).reshape(1, 3)
        assert_allclose(index, index_th)

    def test_find_inside_points(self):
        th_boxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 4)
        boxes = HorizontalBoxes(th_boxes)
        points = torch.Tensor([[0, 0], [0, 15], [15, 0], [15, 15]])
        index = boxes.find_inside_points(points)
        index_th = torch.BoolTensor([False, False, False, True]).reshape(4, 1)
        assert_allclose(index, index_th)
        # is_aligned
        boxes = boxes.expand(4, 4)
        index = boxes.find_inside_points(points, is_aligned=True)
        index_th = torch.BoolTensor([False, False, False, True])
        assert_allclose(index, index_th)

    def test_from_instance_masks(self):
        bitmap_masks = BitmapMasks.random()
        boxes = HorizontalBoxes.from_instance_masks(bitmap_masks)
        self.assertIsInstance(boxes, HorizontalBoxes)
        self.assertEqual(len(boxes), len(bitmap_masks))
        polygon_masks = PolygonMasks.random()
        boxes = HorizontalBoxes.from_instance_masks(polygon_masks)
        self.assertIsInstance(boxes, HorizontalBoxes)
        self.assertEqual(len(boxes), len(bitmap_masks))
        # zero length masks
        bitmap_masks = BitmapMasks.random(num_masks=0)
        boxes = HorizontalBoxes.from_instance_masks(bitmap_masks)
        self.assertIsInstance(boxes, HorizontalBoxes)
        self.assertEqual(len(boxes), 0)
        polygon_masks = PolygonMasks.random(num_masks=0)
        boxes = HorizontalBoxes.from_instance_masks(polygon_masks)
        self.assertIsInstance(boxes, HorizontalBoxes)
        self.assertEqual(len(boxes), 0)
