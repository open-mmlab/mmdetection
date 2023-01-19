# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.testing import get_detector_cfg
from mmdet.utils import register_all_modules


class TestConditionalDETR(TestCase):

    def setUp(self) -> None:
        register_all_modules()

    def test_conditional_detr_head_loss(self):
        """Tests transformer head loss when truth is empty and non-empty."""
        s = 256
        metainfo = {
            'img_shape': (s, s),
            'scale_factor': (1, 1),
            'pad_shape': (s, s),
            'batch_input_shape': (s, s)
        }
        img_metas = DetDataSample()
        img_metas.set_metainfo(metainfo)
        batch_data_samples = []
        batch_data_samples.append(img_metas)

        config = get_detector_cfg(
            'conditional_detr/conditional-detr_r50_8xb2-50e_coco.py')

        model = MODELS.build(config)
        model.init_weights()
        random_image = torch.rand(1, 3, s, s)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        img_metas.gt_instances = gt_instances
        batch_data_samples1 = []
        batch_data_samples1.append(img_metas)
        empty_gt_losses = model.loss(
            random_image, batch_data_samples=batch_data_samples1)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        for key, loss in empty_gt_losses.items():
            if 'cls' in key:
                self.assertGreater(loss.item(), 0,
                                   'cls loss should be non-zero')
            elif 'bbox' in key:
                self.assertEqual(
                    loss.item(), 0,
                    'there should be no box loss when no ground true boxes')
            elif 'iou' in key:
                self.assertEqual(
                    loss.item(), 0,
                    'there should be no iou loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        img_metas.gt_instances = gt_instances
        batch_data_samples2 = []
        batch_data_samples2.append(img_metas)
        one_gt_losses = model.loss(
            random_image, batch_data_samples=batch_data_samples2)
        for loss in one_gt_losses.values():
            self.assertGreater(
                loss.item(), 0,
                'cls loss, or box loss, or iou loss should be non-zero')

        model.eval()
        # test _forward
        model._forward(random_image, batch_data_samples=batch_data_samples2)
        # test only predict
        model.predict(
            random_image, batch_data_samples=batch_data_samples2, rescale=True)
