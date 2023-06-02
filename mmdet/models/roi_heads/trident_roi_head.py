# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class TridentRoIHead(StandardRoIHead):
    """Trident roi head.

    Args:
        num_branch (int): Number of branches in TridentNet.
        test_branch_idx (int): In inference, all 3 branches will be used
            if `test_branch_idx==-1`, otherwise only branch with index
            `test_branch_idx` will be used.
    """

    def __init__(self, num_branch: int, test_branch_idx: int,
                 **kwargs) -> None:
        self.num_branch = num_branch
        self.test_branch_idx = test_branch_idx
        super().__init__(**kwargs)

    def merge_trident_bboxes(self,
                             trident_results: InstanceList) -> InstanceData:
        """Merge bbox predictions of each branch.

        Args:
            trident_results (List[:obj:`InstanceData`]): A list of InstanceData
                predicted from every branch.

        Returns:
            :obj:`InstanceData`: merged InstanceData.
        """
        bboxes = torch.cat([res.bboxes for res in trident_results])
        scores = torch.cat([res.scores for res in trident_results])
        labels = torch.cat([res.labels for res in trident_results])

        nms_cfg = self.test_cfg['nms']
        results = InstanceData()
        if bboxes.numel() == 0:
            results.bboxes = bboxes
            results.scores = scores
            results.labels = labels
        else:
            det_bboxes, keep = batched_nms(bboxes, scores, labels, nms_cfg)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = labels[keep]

        if self.test_cfg['max_per_img'] > 0:
            results = results[:self.test_cfg['max_per_img']]
        return results

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        - Compute prediction bbox and label per branch.
        - Merge predictions of each branch according to scores of
          bboxes, i.e., bboxes with higher score are kept to give
          top-k prediction.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results to
                the original image. Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results_list = super().predict(
            x=x,
            rpn_results_list=rpn_results_list,
            batch_data_samples=batch_data_samples,
            rescale=rescale)

        num_branch = self.num_branch \
            if self.training or self.test_branch_idx == -1 else 1

        merged_results_list = []
        for i in range(len(batch_data_samples) // num_branch):
            merged_results_list.append(
                self.merge_trident_bboxes(results_list[i * num_branch:(i + 1) *
                                                       num_branch]))
        return merged_results_list
