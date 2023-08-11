# Copyright (c) OpenMMLab. All rights reserved.
import json
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.testing import get_detector_cfg
from mmdet.utils import register_all_modules


class TestDDQDETR(TestCase):

    def setUp(self):
        register_all_modules()

    def test_ddq_detr_head_loss(self):
        """Tests DDQ DETR head loss when truth is empty and non-empty."""
        configs = [
            get_detector_cfg('ddq_detr/ddq-detr-4scale_r50_8xb2-12e_coco.py')
        ]  # noqa E501

        for config in configs:
            model = MODELS.build(config)
            model.init_weights()

            batch_data_samples = self.get_batch_data_samples()

            random_images = torch.rand([1, 3, 800, 1067])

            # Test that empty ground truth encourages the network to
            # predict background
            gt_instances = InstanceData()
            gt_instances.bboxes = torch.empty((0, 4))
            gt_instances.labels = torch.LongTensor([])
            data_sample = DetDataSample()
            data_sample.set_metainfo(batch_data_samples[0].metainfo)
            data_sample.gt_instances = gt_instances

            batch_data_samples_1 = [data_sample]
            empty_gt_losses = model.loss(
                random_images, batch_data_samples=batch_data_samples_1)
            # When there is no truth, the cls loss should be nonzero but there
            # should be no box or aux loss.
            zero_loss_keywords = ['bbox', 'iou', 'dn', 'aux']

            for key, loss in empty_gt_losses.items():
                _loss = loss.item()
                if any(zero_loss_keyword in key
                       for zero_loss_keyword in zero_loss_keywords):
                    self.assertEqual(
                        _loss, 0, f'there should be no {key}({_loss}) '
                        f'when no ground true boxes')
                elif 'cls' in key:
                    self.assertGreater(_loss, 0,
                                       f'{key}({_loss}) should be non-zero')

            # When truth is non-empty then both cls and box loss should
            # be nonzero for random inputs.
            batch_data_samples_2 = batch_data_samples
            one_gt_losses = model.loss(
                random_images, batch_data_samples=batch_data_samples_2)
            for loss in one_gt_losses.values():
                self.assertGreater(
                    loss.item(), 0,
                    'cls loss, or box loss, or iou loss should be non-zero')

            model.eval()
            # test _forward
            model._forward(
                random_images, batch_data_samples=batch_data_samples_2)
            # test only predict
            model.predict(
                random_images,
                batch_data_samples=batch_data_samples_2,
                rescale=True)

    def get_batch_data_samples(self):
        """Generate batch data samples including model inputs and gt labels."""
        data_sample_file_path = 'tests/data/coco_batched_sample.json'

        with open(data_sample_file_path, 'r') as file_stream:
            data_sample_infos = json.load(file_stream)

        batch_data_samples = []

        for data_sample_info in data_sample_infos:
            data_sample = DetDataSample()

            metainfo = data_sample_info['metainfo']
            labels = data_sample_info['labels']
            bboxes = data_sample_info['bboxes']

            data_sample.set_metainfo(metainfo)

            gt_instances = InstanceData()
            gt_instances.labels = torch.LongTensor(labels)
            gt_instances.bboxes = torch.Tensor(bboxes)
            data_sample.gt_instances = gt_instances

            batch_data_samples.append(data_sample)

        return batch_data_samples
