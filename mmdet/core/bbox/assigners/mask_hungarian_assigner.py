import numpy as np
import torch

from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs.builder import MATCH_COST, build_match_cost
from ..match_costs.match_cost import FocalLossCost
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@MATCH_COST.register_module()
class BinaryDiceCost:

    def __init__(self,
                 weight=1.,
                 pred_act=False,
                 eps=1e-3):
        self.weight = weight
        self.pred_act = pred_act
        self.eps = eps

    def binary_mask_dice_loss(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): shape = [N1, H, W]
            gt_masks (Tensor): shape = [N2, H, W], store 0 or 1, 0 for negative class
                and 1 for postive class

        Returns:
            Tensor: shape = [N1, N2]
        """
        mask_preds = mask_preds.reshape((mask_preds.shape[0], -1))
        gt_masks = gt_masks.reshape((gt_masks.shape[0], -1)).float()
        numerator = 2 * torch.einsum("nc,mc->nm", mask_preds, gt_masks)
        denominator = mask_preds.sum(-1)[:, None] + gt_masks.sum(-1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)        
        return loss

    def __call__(self, mask_preds, gt_masks):
        if self.pred_act:
            mask_preds = mask_preds.sigmoid()
        dice_cost = self.binary_mask_dice_loss(mask_preds, gt_masks)
        return dice_cost * self.weight


@MATCH_COST.register_module()
class MaskFocalLossCost(FocalLossCost):

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): shape [num_query, h, w], dtype=torch.float32
            gt_labels (Tensor): shape [num_gts, h, w], dtype=torch.long

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.reshape((cls_pred.shape[0], -1))
        gt_labels = gt_labels.reshape((gt_labels.shape[0], -1)).float()
        hw = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        
        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels)) 
        return cls_cost / hw * self.weight


@BBOX_ASSIGNERS.register_module()
class MaskHungarianAssigner(BaseAssigner):

    def __init__(self, 
                cls_cost, 
                mask_cost, 
                dice_cost):
        self.cls_cost = build_match_cost(cls_cost)
        self.mask_cost = build_match_cost(mask_cost)
        self.dice_cost = build_match_cost(dice_cost)

    def assign(self, 
               cls_pred, 
               mask_pred, 
               gt_labels, 
               gt_mask, 
               img_meta, 
               gt_bboxes_ignore=None, 
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        Args:
            cls_pred (Tensor): shape = [N1, ]
            mask_pred (Tensor): shape = [N1, H, W]
            gt_labels (Tensor): shape = [N2, ]
            gt_mask (Tensor): shape = [N2, H, W]
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            (:obj:`AssignResult`): The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_queries = gt_labels.shape[0], cls_pred.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = cls_pred.new_full((num_queries, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = cls_pred.new_full((num_queries, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_queries == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and maskcost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0
        
        if self.mask_cost.weight != 0:
            # mask_pred shape = [nq, h, w]
            # gt_mask shape = [ng, h, w]
            # mask_cost shape = [nq, ng]
            mask_cost = self.mask_cost(mask_pred, gt_mask)
        else:
            mask_cost = 0
        
        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(mask_pred, gt_mask)
        else:
            dice_cost = 0
        cost = cls_cost + mask_cost + dice_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            cls_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            cls_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
