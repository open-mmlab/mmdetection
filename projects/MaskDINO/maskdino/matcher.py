# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DINO https://github.com/IDEA-Research/DINO by Feng Li and Hao Zhang.
"""Modules to compute the matching cost and solve the corresponding LSAP."""
import torch
import torch.nn.functional as F
from mmcv.ops import point_sample
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

# from torchvision.ops import generalized_box_iou  #TODO


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction='none')

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum(
        'nc,mc->nm', neg, (1 - targets))

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def setup_seed(seed):
    import random

    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


from torchvision.ops.boxes import box_area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)  # diff


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the
    predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of
    this, in general, there are more predictions than targets. In this case, we
    do a 1-to-1 matching of the best predictions, while the others are un-
    matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_mask: float = 1,
                 cost_dice: float = 1,
                 num_points: int = 0,
                 cost_box: float = 0,
                 cost_giou: float = 0,
                 panoptic_on: bool = False):
        """Creates the matcher.

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou

        self.panoptic_on = panoptic_on

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, 'all costs can‘t be 0'

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self,
                                 outputs,
                                 targets,
                                 cost=['cls', 'box', 'mask']):
        """More memory-friendly matching.

        Change cost to compute only certain loss in matching
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_bbox = outputs['pred_boxes'][b]
            if 'box' in cost:
                tgt_bbox = targets[b]['boxes']
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_giou = -generalized_box_iou(
                    bbox_cxcywh_to_xyxy(out_bbox),
                    bbox_cxcywh_to_xyxy(tgt_bbox))
            else:
                cost_bbox = torch.tensor(0).to(out_bbox)
                cost_giou = torch.tensor(0).to(out_bbox)

            out_prob = outputs['pred_logits'][b].sigmoid(
            )  # [num_queries, num_classes]
            tgt_ids = targets[b]['labels']
            # focal loss
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (
                -(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * (
                (1 - out_prob)**gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:,
                                                                     tgt_ids]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be omitted.
            # cost_class = -out_prob[:, tgt_ids]
            if 'mask' in cost:
                out_mask = outputs['pred_masks'][
                    b]  # [num_queries, H_pred, W_pred]
                # gt masks are already padded when preparing target
                tgt_mask = targets[b]['masks'].to(out_mask)

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                # setup_seed(20)
                point_coords = torch.rand(
                    1, self.num_points, 2, device=out_mask.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # If there's no annotations
                    if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                        # Compute the focal loss between masks
                        cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                        # Compute the dice loss between masks
                        cost_dice = batch_dice_loss(out_mask, tgt_mask)
                    else:
                        cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                        cost_dice = batch_dice_loss(out_mask, tgt_mask)

            else:
                cost_mask = torch.tensor(0).to(out_bbox)
                cost_dice = torch.tensor(0).to(out_bbox)

            # Final cost matrix
            if self.panoptic_on:
                isthing = tgt_ids < 80
                cost_bbox[:, ~isthing] = cost_bbox[:, isthing].mean()
                cost_giou[:, ~isthing] = cost_giou[:, isthing].mean()
                cost_bbox[cost_bbox.isnan()] = 0.0
                cost_giou[cost_giou.isnan()] = 0.0

            C = (
                self.cost_mask * cost_mask + self.cost_class * cost_class +
                self.cost_dice * cost_dice + self.cost_box * cost_bbox +
                self.cost_giou * cost_giou)
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets, cost=['cls', 'box', 'mask']):
        """Performs the matching.

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, cost)

    def __repr__(self, _repr_indent=4):
        head = 'Matcher ' + self.__class__.__name__
        body = [
            'cost_class: {}'.format(self.cost_class),
            'cost_mask: {}'.format(self.cost_mask),
            'cost_dice: {}'.format(self.cost_dice),
        ]
        lines = [head] + [' ' * _repr_indent + line for line in body]
        return '\n'.join(lines)
