# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import reduce_mean


def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (inputs * inputs).sum(-1)[:,
                                            None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


@MODELS.register_module()
class SparseInstCriterion(nn.Module):
    """This part is partially derivated from:

    https://github.com/facebookresearch/detr/blob/main/models/detr.py.
    """

    def __init__(
        self,
        num_classes,
        assigner,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            alpha=0.25,
            gamma=2.0,
            reduction='sum',
            loss_weight=2.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            reduction='sum',
            eps=5e-5,
            loss_weight=2.0),
    ):
        super().__init__()
        self.matcher = TASK_UTILS.build(assigner)
        self.num_classes = num_classes
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_obj = MODELS.build(loss_obj)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_classification(self, outputs, batch_gt_instances, indices,
                            num_instances):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [gt.labels[J] for gt, (_, J) in zip(batch_gt_instances, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        target_classes = target_classes.flatten(0, 1)
        # comp focal loss.
        class_loss = self.loss_cls(
            src_logits,
            target_classes,
        ) / num_instances
        return class_loss

    def loss_masks_with_iou_objectness(self, outputs, batch_gt_instances,
                                       indices, num_instances):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # Bx100xHxW
        assert 'pred_masks' in outputs
        assert 'pred_scores' in outputs
        src_iou_scores = outputs['pred_scores']
        src_masks = outputs['pred_masks']
        with torch.no_grad():
            target_masks = torch.cat([
                gt.masks.to_tensor(
                    dtype=src_masks.dtype, device=src_masks.device)
                for gt in batch_gt_instances
            ])
        num_masks = [len(gt.masks) for gt in batch_gt_instances]
        target_masks = target_masks.to(src_masks)
        if len(target_masks) == 0:

            loss_dice = src_masks.sum() * 0.0
            loss_mask = src_masks.sum() * 0.0
            loss_objectness = src_iou_scores.sum() * 0.0

            return loss_objectness, loss_dice, loss_mask

        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(
            target_masks[:, None],
            size=src_masks.shape[-2:],
            mode='bilinear',
            align_corners=False).squeeze(1)

        src_masks = src_masks.flatten(1)
        # FIXME: tgt_idx
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])
        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum:cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx].flatten(1)

        with torch.no_grad():
            ious = compute_mask_iou(src_masks, target_masks)

        tgt_iou_scores = ious
        src_iou_scores = src_iou_scores[src_idx]
        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)

        loss_objectness = self.loss_obj(src_iou_scores, tgt_iou_scores)
        loss_dice = self.loss_dice(src_masks, target_masks) / num_instances
        loss_mask = self.loss_mask(src_masks, target_masks)

        return loss_objectness, loss_dice, loss_mask

    def forward(self, outputs, batch_gt_instances, batch_img_metas,
                batch_gt_instances_ignore):
        # Retrieve the matching between the outputs of
        # the last layer and the targets
        indices = self.matcher(outputs, batch_gt_instances)
        # Compute the average number of target boxes
        # across all nodes, for normalization purposes
        num_instances = sum(gt.labels.shape[0] for gt in batch_gt_instances)
        num_instances = torch.as_tensor([num_instances],
                                        dtype=torch.float,
                                        device=next(iter(
                                            outputs.values())).device)
        num_instances = reduce_mean(num_instances).clamp_(min=1).item()
        # Compute all the requested losses
        loss_cls = self.loss_classification(outputs, batch_gt_instances,
                                            indices, num_instances)
        loss_obj, loss_dice, loss_mask = self.loss_masks_with_iou_objectness(
            outputs, batch_gt_instances, indices, num_instances)

        return dict(
            loss_cls=loss_cls,
            loss_obj=loss_obj,
            loss_dice=loss_dice,
            loss_mask=loss_mask)


@TASK_UTILS.register_module()
class SparseInstMatcher(nn.Module):

    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mask_score = dice_score

    def forward(self, outputs, batch_gt_instances):
        with torch.no_grad():
            B, N, H, W = outputs['pred_masks'].shape
            pred_masks = outputs['pred_masks']
            pred_logits = outputs['pred_logits'].sigmoid()
            device = pred_masks.device

            tgt_ids = torch.cat([gt.labels for gt in batch_gt_instances])

            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(pred_logits),
                         torch.as_tensor([]).to(pred_logits))] * B
            tgt_masks = torch.cat([
                gt.masks.to_tensor(dtype=pred_masks.dtype, device=device)
                for gt in batch_gt_instances
            ])

            tgt_masks = F.interpolate(
                tgt_masks[:, None],
                size=pred_masks.shape[-2:],
                mode='bilinear',
                align_corners=False).squeeze(1)

            pred_masks = pred_masks.view(B * N, -1)
            tgt_masks = tgt_masks.flatten(1)
            with autocast(enabled=False):
                pred_masks = pred_masks.float()
                tgt_masks = tgt_masks.float()
                pred_logits = pred_logits.float()
                mask_score = self.mask_score(pred_masks, tgt_masks)
                # Nx(Number of gts)
                matching_prob = pred_logits.view(B * N, -1)[:, tgt_ids]
                C = (mask_score**self.alpha) * (matching_prob**self.beta)

            C = C.view(B, N, -1).cpu()
            # hungarian matching
            sizes = [len(gt.masks) for gt in batch_gt_instances]
            indices = [
                linear_sum_assignment(c[i], maximize=True)
                for i, c in enumerate(C.split(sizes, -1))
            ]
            indices = [(torch.as_tensor(i, dtype=torch.int64),
                        torch.as_tensor(j, dtype=torch.int64))
                       for i, j in indices]
            return indices
