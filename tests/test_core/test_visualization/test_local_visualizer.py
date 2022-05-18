import os
from unittest import TestCase

import cv2
import numpy as np
import torch
from mmengine.data import InstanceData

from mmdet.core import DetDataSample
from mmdet.core.visualization import DetLocalVisualizer


def _rand_bboxes(num_boxes, h, w):
    cx, cy, bw, bh = torch.rand(num_boxes, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clip(0, w)
    tl_y = ((cy * h) - (h * bh / 2)).clip(0, h)
    br_x = ((cx * w) + (w * bw / 2)).clip(0, w)
    br_y = ((cy * h) + (h * bh / 2)).clip(0, h)

    bboxes = torch.vstack([tl_x, tl_y, br_x, br_y]).T
    return bboxes


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
        gt_instances.labels = torch.randint(0, num_class, (5, ))
        gt_det_data_sample = DetDataSample()
        gt_det_data_sample.gt_instances = gt_instances

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
        pred_instances.labels = torch.randint(0, num_class, (5, ))
        pred_instances.scores = torch.rand((5, ))
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

        # TODO: test gt_panoptic_seg

    def _assert_image_and_shape(self, out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape
        os.remove(out_file)
