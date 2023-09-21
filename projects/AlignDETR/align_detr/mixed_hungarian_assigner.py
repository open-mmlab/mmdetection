# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmdet.models.task_modules import AssignResult, BaseAssigner
from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class MixedHungarianAssigner(BaseAssigner):
    """Computes 1-to-k matching between ground truth and predictions.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the 1-to-k
    gt-pred matching, the un-matched are treated as backgrounds. Thus
    each query prediction will be assigned with `0` or a positive integer
    indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
    """

    def __init__(
        self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict]
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               k: int = 1,
               **kwargs) -> AssignResult:
        """Computes 1-to-k gt-pred matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. Assign every prediction to -1.
        2. Compute the weighted costs, each cost has shape
            (num_preds, num_gts).
        3. Update k according to num_preds and num_gts, then repeat
            costs k times to shape: (num_preds, k * num_gts), so that each
            gt will match k predictions.
        4. Do Hungarian matching on CPU based on the costs.
        5. Assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places. It may includes ``masks``, with shape
                (n, h, w) or (n, l).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                ``labels``, with shape (k, ) and ``masks``, with shape
                (k, h, w) or (k, l).
            img_meta (dict): Image information for one image.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. Assign -1 by default.
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment.
            if num_gts == 0:
                # No ground truth, assign all to background.
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. Compute weighted costs.
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)

        # 3. Update k according to num_preds and num_gts,  then
        #   repeat the ground truth k times to perform 1-to-k gt-pred
        #   matching. For example, if num_preds = 900, num_gts = 3, then
        #   there are only 3 gt-pred pairs in sum for 1-1 matching.
        #   However, for 1-k gt-pred matching, if k = 4, then each
        #   gt is assigned 4 unique predictions, so there would be 12
        #   gt-pred pairs in sum.
        k = max(1, min(k, num_preds // num_gts))
        cost = cost.repeat(1, k)

        # 4. Do Hungarian matching on CPU using linear_sum_assignment.
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        matched_col_inds = matched_col_inds % num_gts
        # 5. Assign backgrounds and foregrounds.
        # Assign all indices to backgrounds first.
        assigned_gt_inds[:] = 0
        # Assign foregrounds based on matching results.
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        assign_result = AssignResult(
            num_gts=k * num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)

        return assign_result
