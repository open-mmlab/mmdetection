import unittest

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmengine.testing import assert_allclose

from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.models.seg_heads.panoptic_fusion_heads import HeuristicFusionHead


class TestHeuristicFusionHead(unittest.TestCase):

    def test_loss(self):
        head = HeuristicFusionHead(num_things_classes=2, num_stuff_classes=2)
        result = head.loss()
        self.assertTrue(not head.with_loss)
        self.assertDictEqual(result, dict())

    def test_predict(self):
        test_cfg = Config(dict(mask_overlap=0.5, stuff_area_limit=1))
        head = HeuristicFusionHead(
            num_things_classes=2, num_stuff_classes=2, test_cfg=test_cfg)
        mask_results = InstanceData()
        mask_results.bboxes = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]])
        mask_results.labels = torch.tensor([0, 1])
        mask_results.scores = torch.tensor([0.8, 0.7])
        mask_results.masks = torch.tensor([[[1, 0], [0, 0]], [[0, 0],
                                                              [0, 1]]]).bool()

        seg_preds_list = [
            torch.tensor([[[0.2, 0.7], [0.3, 0.1]], [[0.2, 0.2], [0.6, 0.1]],
                          [[0.6, 0.1], [0.1, 0.8]]])
        ]
        target_list = [
            torch.tensor([[0 + 1 * INSTANCE_OFFSET, 2],
                          [3, 1 + 2 * INSTANCE_OFFSET]])
        ]
        results_list = head.predict([mask_results], seg_preds_list)
        for target, result in zip(target_list, results_list):
            assert_allclose(result.sem_seg[0], target)

        # test with no thing
        head = HeuristicFusionHead(
            num_things_classes=2, num_stuff_classes=2, test_cfg=test_cfg)
        mask_results = InstanceData()
        mask_results.bboxes = torch.zeros((0, 4))
        mask_results.labels = torch.zeros((0, )).long()
        mask_results.scores = torch.zeros((0, ))
        mask_results.masks = torch.zeros((0, 2, 2), dtype=torch.bool)
        seg_preds_list = [
            torch.tensor([[[0.2, 0.7], [0.3, 0.1]], [[0.2, 0.2], [0.6, 0.1]],
                          [[0.6, 0.1], [0.1, 0.8]]])
        ]
        target_list = [torch.tensor([[4, 2], [3, 4]])]

        results_list = head.predict([mask_results], seg_preds_list)
        for target, result in zip(target_list, results_list):
            assert_allclose(result.sem_seg[0], target)
