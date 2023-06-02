# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch
from mmengine.fileio import load
from torch import Tensor

from mmdet.evaluation import DumpDetResults
from mmdet.structures.mask import encode_mask_results


class TestDumpResults(TestCase):

    def test_init(self):
        with self.assertRaisesRegex(ValueError,
                                    'The output file must be a pkl file.'):
            DumpDetResults(out_file_path='./results.json')

    def test_process(self):
        metric = DumpDetResults(out_file_path='./results.pkl')
        data_samples = [dict(data=(Tensor([1, 2, 3]), Tensor([4, 5, 6])))]
        metric.process(None, data_samples)
        self.assertEqual(len(metric.results), 1)
        self.assertEqual(metric.results[0]['data'][0].device,
                         torch.device('cpu'))

        metric = DumpDetResults(out_file_path='./results.pkl')
        masks = torch.zeros(10, 10, 4)
        data_samples = [
            dict(pred_instances=dict(masks=masks), gt_instances=[])
        ]
        metric.process(None, data_samples)
        self.assertEqual(len(metric.results), 1)
        self.assertEqual(metric.results[0]['pred_instances']['masks'],
                         encode_mask_results(masks.numpy()))
        self.assertNotIn('gt_instances', metric.results[0])

    def test_compute_metrics(self):
        temp_dir = tempfile.TemporaryDirectory()
        path = osp.join(temp_dir.name, 'results.pkl')
        metric = DumpDetResults(out_file_path=path)
        data_samples = [dict(data=(Tensor([1, 2, 3]), Tensor([4, 5, 6])))]
        metric.process(None, data_samples)
        metric.compute_metrics(metric.results)
        self.assertTrue(osp.isfile(path))

        results = load(path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['data'][0].device, torch.device('cpu'))

        temp_dir.cleanup()
