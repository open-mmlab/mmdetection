from unittest import TestCase

import numpy as np
from mmengine.testing import assert_allclose

from mmdet.structures.mask import BitmapMasks, PolygonMasks


class TestMaskStructures(TestCase):

    def test_bitmap_translate_same_size(self):
        mask_array = np.zeros((5, 10, 10), dtype=np.uint8)
        mask_array[:, 0:5, 0:5] = 1
        mask_target = np.zeros((5, 10, 10), dtype=np.uint8)
        mask_target[:, 0:5, 5:10] = 1

        mask = BitmapMasks(mask_array, 10, 10)
        mask = mask.translate((10, 10), 5)
        assert mask.masks.shape == (5, 10, 10)
        assert_allclose(mask_target, mask.masks)

    def test_bitmap_translate_diff_size(self):
        # test out shape larger
        mask_array = np.zeros((5, 10, 10), dtype=np.uint8)
        mask_array[:, 0:5, 0:5] = 1

        mask_target = np.zeros((5, 20, 20), dtype=np.uint8)
        mask_target[:, 0:5, 5:10] = 1
        mask = BitmapMasks(mask_array, 10, 10)
        mask = mask.translate((20, 20), 5)
        assert mask.masks.shape == (5, 20, 20)
        assert_allclose(mask_target, mask.masks)

        # test out shape smaller
        mask_array = np.zeros((5, 10, 10), dtype=np.uint8)
        mask_array[:, 0:5, 0:5] = 1

        mask_target = np.zeros((5, 20, 8), dtype=np.uint8)
        mask_target[:, 0:5, 5:] = 1
        mask = BitmapMasks(mask_array, 10, 10)
        mask = mask.translate((20, 8), 5)
        assert mask.masks.shape == (5, 20, 8)
        assert_allclose(mask_target, mask.masks)

    def test_bitmap_cat(self):
        # test invalid inputs
        with self.assertRaises(AssertionError):
            BitmapMasks.cat(BitmapMasks.random(4))
        with self.assertRaises(ValueError):
            BitmapMasks.cat([])
        with self.assertRaises(AssertionError):
            BitmapMasks.cat([BitmapMasks.random(2), PolygonMasks.random(3)])

        masks = [BitmapMasks.random(num_masks=3) for _ in range(5)]
        cat_mask = BitmapMasks.cat(masks)
        assert len(cat_mask) == 3 * 5
        for i, m in enumerate(masks):
            assert_allclose(m.masks, cat_mask.masks[i * 3:(i + 1) * 3])

    def test_polygon_cat(self):
        # test invalid inputs
        with self.assertRaises(AssertionError):
            PolygonMasks.cat(PolygonMasks.random(4))
        with self.assertRaises(ValueError):
            PolygonMasks.cat([])
        with self.assertRaises(AssertionError):
            PolygonMasks.cat([BitmapMasks.random(2), PolygonMasks.random(3)])

        masks = [PolygonMasks.random(num_masks=3) for _ in range(5)]
        cat_mask = PolygonMasks.cat(masks)
        assert len(cat_mask) == 3 * 5
        for i, m in enumerate(masks):
            assert_allclose(m.masks, cat_mask.masks[i * 3:(i + 1) * 3])
