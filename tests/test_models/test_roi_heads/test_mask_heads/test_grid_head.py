# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from parameterized import parameterized

from mmdet.models.roi_heads.mask_heads import GridHead
from mmdet.models.utils import unpack_gt_instances
from mmdet.testing import (demo_mm_inputs, demo_mm_proposals,
                           demo_mm_sampling_results)


class TestGridHead(TestCase):

    @parameterized.expand(['cpu', 'cuda'])
    def test_grid_head_loss(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        grid_head = GridHead()
        grid_head.to(device=device)

        s = 256
        image_shapes = [(3, s, s)]
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[1],
            num_classes=4,
            with_mask=True,
            device=device)['data_samples']
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100, device=device)

        train_cfg = ConfigDict(dict(pos_radius=1))

        # prepare ground truth
        (batch_gt_instances, batch_gt_instances_ignore,
         _) = unpack_gt_instances(batch_data_samples)
        sampling_results = demo_mm_sampling_results(
            proposals_list=proposals_list,
            batch_gt_instances=batch_gt_instances,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        # prepare grid feats
        pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results])
        grid_feats = torch.rand((pos_bboxes.size(0), 256, 14, 14)).to(device)
        sample_idx = torch.arange(0, pos_bboxes.size(0))
        grid_pred = grid_head(grid_feats)

        grid_head.loss(grid_pred, sample_idx, sampling_results, train_cfg)

    @parameterized.expand(['cpu', 'cuda'])
    def test_mask_iou_head_predict_by_feat(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        grid_head = GridHead()
        grid_head.to(device=device)

        s = 128
        num_samples = 2
        num_classes = 4
        img_metas = {
            'img_shape': (s, s, 3),
            'scale_factor': (1, 1),
            'ori_shape': (s, s, 3)
        }
        results = InstanceData(metainfo=img_metas)
        results.bboxes = torch.rand((num_samples, 4)).to(device)
        results.scores = torch.rand((num_samples, )).to(device)
        results.labels = torch.randint(
            num_classes, (num_samples, ), dtype=torch.long).to(device)

        grid_feats = torch.rand((num_samples, 256, 14, 14)).to(device)
        grid_preds = grid_head(grid_feats)
        grid_head.predict_by_feat(
            grid_preds=grid_preds,
            results_list=[results],
            batch_img_metas=[img_metas])
