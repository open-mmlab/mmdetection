# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.data import InstanceData
from parameterized import parameterized

from mmdet.models.roi_heads.mask_heads import FCNMaskHead


class TestFCNMaskHead(TestCase):

    def test_init(self):
        """Test init standard RoI head."""

    @parameterized.expand(['cpu', 'cuda'])
    def test_get_seg_masks(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')
        num_classes = 6
        mask_head = FCNMaskHead(
            num_convs=1,
            in_channels=1,
            conv_out_channels=1,
            num_classes=num_classes)
        rcnn_test_cfg = ConfigDict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)
        s = 128
        img_metas = {
            'img_shape': (s, s, 3),
            'scale_factor': (1, 1),
            'ori_shape': (s, s, 3)
        }
        result = InstanceData(metainfo=img_metas)

        num_samples = 2
        mask_pred = [torch.rand((num_samples, num_classes, 14, 14)).to(device)]
        result.bboxes = torch.rand((num_samples, 4)).to(device)
        result.labels = torch.randint(
            num_classes, (num_samples, ), dtype=torch.long).to(device)
        mask_head.to(device=device)
        result_list = mask_head.get_results(
            mask_preds=tuple(mask_pred),
            results_list=tuple([result]),
            batch_img_metas=[img_metas],
            rcnn_test_cfg=rcnn_test_cfg)

        assert isinstance(result_list[0], InstanceData)
        assert len(result_list[0]) == num_samples
        assert result_list[0].masks.shape == (num_samples, s, s)

        # num_samples is 0
        num_samples = 0
        result = InstanceData(metainfo=img_metas)
        mask_pred = [torch.rand((num_samples, num_classes, 14, 14)).to(device)]
        result.bboxes = torch.zeros((num_samples, 4)).to(device)
        result.labels = torch.zeros((num_samples, )).to(device)
        result_list = mask_head.get_results(
            mask_preds=tuple(mask_pred),
            results_list=tuple([result]),
            batch_img_metas=[img_metas],
            rcnn_test_cfg=rcnn_test_cfg)

        assert isinstance(result_list[0], InstanceData)
        assert len(result_list[0]) == num_samples
        assert result_list[0].masks.shape == (num_samples, s, s)
