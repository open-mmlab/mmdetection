# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.testing import get_detector_cfg
from mmdet.utils import register_all_modules


class TestDINO(TestCase):

    def setUp(self):
        register_all_modules()

    def test_dino_head_loss(self):
        """Tests transformer head loss when truth is empty and non-empty."""
        s = 256
        metainfo = {
            'img_shape': (s, s),
            'scale_factor': (1, 1),
            'pad_shape': (s, s),
            'batch_input_shape': (s, s)
        }
        data_sample = DetDataSample()
        data_sample.set_metainfo(metainfo)

        configs = [get_detector_cfg('dino/dino-4scale_r50_8xb2-12e_coco.py')]

        for config in configs:
            model = MODELS.build(config)
            model.init_weights()
            random_image = torch.rand(1, 3, s, s)

            # Test that empty ground truth encourages the network to
            # predict background
            gt_instances = InstanceData()
            gt_instances.bboxes = torch.empty((0, 4))
            gt_instances.labels = torch.LongTensor([])
            data_sample.gt_instances = gt_instances
            batch_data_samples_1 = [data_sample]
            empty_gt_losses = model.loss(
                random_image, batch_data_samples=batch_data_samples_1)
            # When there is no truth, the cls loss should be nonzero but there
            # should be no box loss.
            for key, loss in empty_gt_losses.items():
                _loss = loss.item()
                if 'bbox' in key or 'iou' in key or 'dn' in key:
                    self.assertEqual(
                        _loss, 0, f'there should be no {key}({_loss}) '
                        f'when no ground true boxes')
                elif 'cls' in key:
                    self.assertGreater(_loss, 0,
                                       f'{key}({_loss}) should be non-zero')

            # When truth is non-empty then both cls and box loss should
            # be nonzero for random inputs
            gt_instances = InstanceData()
            gt_instances.bboxes = torch.Tensor(
                [[23.6667, 23.8757, 238.6326, 151.8874]])
            gt_instances.labels = torch.LongTensor([2])
            data_sample.gt_instances = gt_instances
            batch_data_samples_2 = [data_sample]
            one_gt_losses = model.loss(
                random_image, batch_data_samples=batch_data_samples_2)
            for loss in one_gt_losses.values():
                self.assertGreater(
                    loss.item(), 0,
                    'cls loss, or box loss, or iou loss should be non-zero')

            model.eval()
            # test _forward
            model._forward(
                random_image, batch_data_samples=batch_data_samples_2)
            # test only predict
            model.predict(
                random_image,
                batch_data_samples=batch_data_samples_2,
                rescale=True)
