import os
from unittest import TestCase

import cv2
import numpy as np
import torch
from mmengine.data import InstanceData, PixelData

from mmdet.datasets import (CityscapesDataset, CocoDataset,
                            CocoPanopticDataset, VOCDataset)
from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer, get_palette, palette_val


def _rand_bboxes(num_boxes, h, w):
    cx, cy, bw, bh = torch.rand(num_boxes, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clamp(0, w)
    tl_y = ((cy * h) - (h * bh / 2)).clamp(0, h)
    br_x = ((cx * w) + (w * bw / 2)).clamp(0, w)
    br_y = ((cy * h) + (h * bh / 2)).clamp(0, h)

    bboxes = torch.vstack([tl_x, tl_y, br_x, br_y]).T
    return bboxes


def _create_panoptic_data(num_boxes, h, w):
    sem_seg = np.zeros((h, w), dtype=np.int64) + 2
    bboxes = _rand_bboxes(num_boxes, h, w).int()
    labels = torch.randint(2, (num_boxes, ))
    for i in range(num_boxes):
        x, y, w, h = bboxes[i]
        sem_seg[y:y + h, x:x + w] = (i + 1) * INSTANCE_OFFSET + labels[i]

    return sem_seg[None]


class TestDetLocalVisualizer(TestCase):

    def test_add_datasample(self):
        h = 12
        w = 10
        num_class = 3
        num_bboxes = 5
        out_file = 'out_file.jpg'

        image = np.random.randint(0, 256, size=(h, w, 3)).astype('uint8')

        # test gt_instances
        gt_instances = InstanceData()
        gt_instances.bboxes = _rand_bboxes(num_bboxes, h, w)
        gt_instances.labels = torch.randint(0, num_class, (num_bboxes, ))
        gt_det_data_sample = DetDataSample()
        gt_det_data_sample.gt_instances = gt_instances
        #
        det_local_visualizer = DetLocalVisualizer()
        det_local_visualizer.add_datasample('image', image, gt_det_data_sample)

        # test out_file
        det_local_visualizer.add_datasample(
            'image', image, gt_det_data_sample, out_file=out_file)
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == (h, w, 3)
        os.remove(out_file)

        # test gt_instances and pred_instances
        pred_instances = InstanceData()
        pred_instances.bboxes = _rand_bboxes(num_bboxes, h, w)
        pred_instances.labels = torch.randint(0, num_class, (num_bboxes, ))
        pred_instances.scores = torch.rand((num_bboxes, ))
        pred_det_data_sample = DetDataSample()
        pred_det_data_sample.pred_instances = pred_instances

        det_local_visualizer.add_datasample(
            'image',
            image,
            gt_det_data_sample,
            pred_det_data_sample,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2, 3))

        det_local_visualizer.add_datasample(
            'image',
            image,
            gt_det_data_sample,
            pred_det_data_sample,
            draw_gt=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        det_local_visualizer.add_datasample(
            'image',
            image,
            gt_det_data_sample,
            pred_det_data_sample,
            draw_pred=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        # test gt_panoptic_seg and pred_panoptic_seg
        det_local_visualizer.dataset_meta = dict(CLASSES=('1', '2'))
        gt_sem_seg = _create_panoptic_data(num_bboxes, h, w)
        panoptic_seg = PixelData(sem_seg=gt_sem_seg)
        gt_det_data_sample = DetDataSample()
        gt_det_data_sample.gt_panoptic_seg = panoptic_seg

        pred_sem_seg = _create_panoptic_data(num_bboxes, h, w)
        panoptic_seg = PixelData(sem_seg=pred_sem_seg)
        pred_det_data_sample = DetDataSample()
        pred_det_data_sample.pred_panoptic_seg = panoptic_seg
        det_local_visualizer.add_datasample(
            'image',
            image,
            gt_det_data_sample,
            pred_det_data_sample,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2, 3))

        # class information must be provided
        det_local_visualizer.dataset_meta = {}
        with self.assertRaises(AssertionError):
            det_local_visualizer.add_datasample(
                'image',
                image,
                gt_det_data_sample,
                pred_det_data_sample,
                out_file=out_file)

    def _assert_image_and_shape(self, out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape
        os.remove(out_file)

    def test_palette(self):
        assert palette_val([(1, 2, 3)])[0] == (1 / 255, 2 / 255, 3 / 255)

        # test list
        palette = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        palette_ = get_palette(palette, 3)
        for color, color_ in zip(palette, palette_):
            assert color == color_

        # test tuple
        palette = get_palette((1, 2, 3), 3)
        assert len(palette) == 3
        for color in palette:
            assert color == (1, 2, 3)

        # test color str
        palette = get_palette('red', 3)
        assert len(palette) == 3
        for color in palette:
            assert color == (255, 0, 0)

        # test dataset str
        palette = get_palette('coco', len(CocoDataset.METAINFO['CLASSES']))
        assert len(palette) == len(CocoDataset.METAINFO['CLASSES'])
        assert palette[0] == (220, 20, 60)
        palette = get_palette(
            'coco', len(CocoPanopticDataset.METAINFO['CLASSES']))  # noqa
        assert len(palette) == len(CocoPanopticDataset.METAINFO['CLASSES'])
        assert palette[-1] == (250, 141, 255)
        palette = get_palette('voc', len(VOCDataset.METAINFO['CLASSES']))
        assert len(palette) == len(VOCDataset.METAINFO['CLASSES'])
        assert palette[0] == (106, 0, 228)
        palette = get_palette(
            'citys', len(CityscapesDataset.METAINFO['CLASSES']))  # noqa
        assert len(palette) == len(CityscapesDataset.METAINFO['CLASSES'])
        assert palette[0] == (220, 20, 60)

        # test random
        palette1 = get_palette('random', 3)
        palette2 = get_palette(None, 3)
        for color1, color2 in zip(palette1, palette2):
            assert isinstance(color1, tuple)
            assert isinstance(color2, tuple)
            assert color1 == color2
