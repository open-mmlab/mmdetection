# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import time
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.structures import InstanceData

from mmdet.engine.hooks import DetVisualizationHook
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer


def _rand_bboxes(num_boxes, h, w):
    cx, cy, bw, bh = torch.rand(num_boxes, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clamp(0, w)
    tl_y = ((cy * h) - (h * bh / 2)).clamp(0, h)
    br_x = ((cx * w) + (w * bw / 2)).clamp(0, w)
    br_y = ((cy * h) + (h * bh / 2)).clamp(0, h)

    bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=0).T
    return bboxes


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:
        DetLocalVisualizer.get_instance('current_visualizer')

        pred_instances = InstanceData()
        pred_instances.bboxes = _rand_bboxes(5, 10, 12)
        pred_instances.labels = torch.randint(0, 2, (5, ))
        pred_instances.scores = torch.rand((5, ))
        pred_det_data_sample = DetDataSample()
        pred_det_data_sample.set_metainfo({
            'img_path':
            osp.join(osp.dirname(__file__), '../../data/color.jpg')
        })
        pred_det_data_sample.pred_instances = pred_instances
        self.outputs = [pred_det_data_sample] * 2

    def test_after_val_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = DetVisualizationHook()
        hook.after_val_iter(runner, 1, {}, self.outputs)

    def test_after_test_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = DetVisualizationHook(draw=True)
        hook.after_test_iter(runner, 1, {}, self.outputs)
        self.assertEqual(hook._test_index, 2)

        # test
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        test_out_dir = timestamp + '1'
        runner.work_dir = timestamp
        runner.timestamp = '1'
        hook = DetVisualizationHook(draw=False, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 1, {}, self.outputs)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{test_out_dir}'))

        hook = DetVisualizationHook(draw=True, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 1, {}, self.outputs)
        self.assertTrue(osp.exists(f'{timestamp}/1/{test_out_dir}'))
        shutil.rmtree(f'{timestamp}')
