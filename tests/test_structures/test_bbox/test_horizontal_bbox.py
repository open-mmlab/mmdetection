from math import sqrt
from unittest import TestCase

import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmdet.structures.bbox import HoriInstanceBoxes


class TestHoriInstanceBoxes(TestCase):

    def test_init(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        th_bboxes_cxcywh = torch.Tensor([15, 15, 10, 10]).reshape(1, 1, 4)

        bboxes = HoriInstanceBoxes(th_bboxes)
        assert_allclose(bboxes.tensor, th_bboxes)
        bboxes = HoriInstanceBoxes(th_bboxes, pattern='xyxy')
        assert_allclose(bboxes.tensor, th_bboxes)
        bboxes = HoriInstanceBoxes(th_bboxes_cxcywh, pattern='cxcywh')
        assert_allclose(bboxes.tensor, th_bboxes)
        with self.assertRaises(ValueError):
            bboxes = HoriInstanceBoxes(th_bboxes, pattern='invalid')

    def test_cxcywh(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        th_bboxes_cxcywh = torch.Tensor([15, 15, 10, 10]).reshape(1, 1, 4)
        bboxes = HoriInstanceBoxes(th_bboxes)

        assert_allclose(
            HoriInstanceBoxes.xyxy_to_cxcywh(th_bboxes), th_bboxes_cxcywh)
        assert_allclose(th_bboxes,
                        HoriInstanceBoxes.cxcywh_to_xyxy(th_bboxes_cxcywh))
        assert_allclose(bboxes.cxcywh, th_bboxes_cxcywh)

    def test_propoerty(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        bboxes = HoriInstanceBoxes(th_bboxes)

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
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        img_shape = [50, 85]
        bboxes = HoriInstanceBoxes(th_bboxes)

        # horizontal flip
        flipped_bboxes_th = torch.Tensor([65, 10, 75, 20]).reshape(1, 1, 4)
        flipped_bboxes = bboxes.flip(img_shape, direction='horizontal')
        assert_allclose(flipped_bboxes.tensor, flipped_bboxes_th)
        # vertical flip
        flipped_bboxes_th = torch.Tensor([10, 30, 20, 40]).reshape(1, 1, 4)
        flipped_bboxes = bboxes.flip(img_shape, direction='vertical')
        assert_allclose(flipped_bboxes.tensor, flipped_bboxes_th)
        # diagonal flip
        flipped_bboxes_th = torch.Tensor([65, 30, 75, 40]).reshape(1, 1, 4)
        flipped_bboxes = bboxes.flip(img_shape, direction='diagonal')
        assert_allclose(flipped_bboxes.tensor, flipped_bboxes_th)

    def test_translate(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        bboxes = HoriInstanceBoxes(th_bboxes)

        translated_bboxes = bboxes.translate([23, 46])
        translated_bboxes_th = torch.Tensor([33, 56, 43, 66]).reshape(1, 1, 4)
        assert_allclose(translated_bboxes.tensor, translated_bboxes_th)
        # negative
        translated_bboxes = bboxes.translate([-6, -2])
        translated_bboxes_th = torch.Tensor([4, 8, 14, 18]).reshape(1, 1, 4)
        assert_allclose(translated_bboxes.tensor, translated_bboxes_th)

    def test_clip(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        img_shape = [13, 14]
        bboxes = HoriInstanceBoxes(th_bboxes)

        cliped_bboxes = bboxes.clip(img_shape)
        cliped_bboxes_th = torch.Tensor([10, 10, 14, 13]).reshape(1, 1, 4)
        assert_allclose(cliped_bboxes.tensor, cliped_bboxes_th)

    def test_rotate(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        center = (15, 15)
        angle = 45
        bboxes = HoriInstanceBoxes(th_bboxes)

        rotated_bboxes = bboxes.rotate(center, angle)
        rotated_bboxes_th = torch.Tensor([
            15 - 5 * sqrt(2), 15 - 5 * sqrt(2), 15 + 5 * sqrt(2),
            15 + 5 * sqrt(2)
        ]).reshape(1, 1, 4)
        assert_allclose(rotated_bboxes.tensor, rotated_bboxes_th)

    def test_project(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        matrix = np.random.rand(3, 3)
        bboxes = HoriInstanceBoxes(th_bboxes)
        bboxes.project(matrix)

    def test_rescale(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        scale_factor = [0.4, 0.8]
        bboxes = HoriInstanceBoxes(th_bboxes)

        rescaled_bboxes = bboxes.rescale(scale_factor)
        rescaled_bboxes_th = torch.Tensor([4, 8, 8, 16]).reshape(1, 1, 4)
        assert_allclose(rescaled_bboxes.tensor, rescaled_bboxes_th)
        rescaled_bboxes = bboxes.rescale(scale_factor, mapping_back=True)
        rescaled_bboxes_th = torch.Tensor([25, 12.5, 50, 25]).reshape(1, 1, 4)
        assert_allclose(rescaled_bboxes.tensor, rescaled_bboxes_th)

    def test_resize_bboxes(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        scale_factor = [0.4, 0.8]
        bboxes = HoriInstanceBoxes(th_bboxes)

        resized_bboxes = bboxes.resize_bboxes(scale_factor)
        resized_bboxes_th = torch.Tensor([13, 11, 17, 19]).reshape(1, 1, 4)
        assert_allclose(resized_bboxes.tensor, resized_bboxes_th)

    def test_is_bboxes_inside(self):
        th_bboxes = torch.Tensor([[10, 10, 20, 20], [-5, -5, 15, 15],
                                  [45, 45, 55, 55]]).reshape(1, 3, 4)
        img_shape = [30, 30]
        bboxes = HoriInstanceBoxes(th_bboxes)

        index = bboxes.is_bboxes_inside(img_shape)
        index_th = torch.BoolTensor([True, True, False]).reshape(1, 3)
        self.assertEqual(tuple(index.size()), (1, 3))
        assert_allclose(index, index_th)

    def test_find_inside_points(self):
        th_bboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        bboxes = HoriInstanceBoxes(th_bboxes)
        points = torch.Tensor([[0, 0], [0, 15], [15, 0], [15, 15]])
        index = bboxes.find_inside_points(points)
        index_th = torch.BoolTensor([False, False, False, True]).reshape(4, 1)
        self.assertEqual(tuple(index.size()), (4, 1))
        assert_allclose(index, index_th)
