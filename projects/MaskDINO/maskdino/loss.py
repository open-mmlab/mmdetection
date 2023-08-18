# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DINO https://github.com/IDEA-Research/DINO by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
"""MaskFormer criterion."""
import logging
from typing import List

import torch
import torch.nn.functional as F
from mmcv.ops import point_sample
from mmengine.dist import get_world_size
from torch import nn

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from .matcher import HungarianMatcher
from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

# from torchvision.ops import generalized_box_iou


def sigmoid_focal_loss(inputs,
                       targets,
                       num_boxes,
                       alpha: float = 0.25,
                       gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
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
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
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
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction='none')

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """We estimate uncerainty as L1 distance between 0.0 and the logit
    prediction in 'logits' for the foreground class in `classes`.

    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


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


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.

    The process happens in two steps:     1) we compute hungarian assignment
    between ground truth boxes and the outputs of the model     2) we supervise
    each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self,
                 num_classes,
                 matcher,
                 class_weight=4.0,
                 box_weight=5.0,
                 giou_weight=2.0,
                 mask_weight=5.0,
                 dice_weight=5.0,
                 dn='seg',
                 dec_layers=9,
                 box_loss=True,
                 eos_coef=0.1,
                 num_points=100,
                 oversample_ratio=3.0,
                 importance_sample_ratio=0.75,
                 semantic_ce_loss=False,
                 panoptic_on=False,
                 two_stage=True,
                 deep_supervision=True):
        """Create the criterion.

        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(**matcher, panoptic_on=panoptic_on)

        weight_dict, losses, dn_losses = _process_criterion_info(
            class_weight,
            mask_weight,
            dice_weight,
            box_weight,
            giou_weight,
            dn=dn,
            dec_layers=dec_layers,
            box_loss=box_loss,
            two_stage=two_stage,
            deep_supervision=deep_supervision)

        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.dn = dn
        self.dn_losses = dn_losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25

        self.panoptic_on = panoptic_on
        self.semantic_ce_loss = semantic_ce_loss

    def loss_labels_ce(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL) targets dicts must contain the key
        "labels" containing a tensor of dim [nb_target_boxes]"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss) targets dicts must contain
        the key "labels" containing a tensor of dim [nb_target_boxes]"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([
            src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1
        ],
                                            dtype=src_logits.dtype,
                                            layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression
        loss and the GIoU loss targets dicts must contain the key "boxes"
        containing a tensor of dim [nb_target_boxes, 4] The target boxes are
        expected in format (center_x, center_y, w, h), normalized by the image
        size."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                bbox_cxcywh_to_xyxy(src_boxes),
                bbox_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_boxes_panoptic(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression
        loss and the GIoU loss targets dicts must contain the key "boxes"
        containing a tensor of dim [nb_target_boxes, 4] The target boxes are
        expected in format (center_x, center_y, w, h), normalized by the image
        size."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat(
            [t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        isthing = target_labels < 80
        target_boxes = target_boxes[isthing]
        src_boxes = src_boxes[isthing]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                bbox_cxcywh_to_xyxy(src_boxes),
                bbox_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice
        loss.

        targets dicts must contain the key "masks" containing a tensor of dim
        [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            # setup_seed(20)
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            'loss_mask': sigmoid_ce_loss_jit(point_logits, point_labels,
                                             num_masks),
            'loss_dice': dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def prep_for_dn(self, mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']

        known_indice = mask_dict['known_indice']
        scalar, pad_size = mask_dict['scalar'], mask_dict['pad_size']
        assert pad_size % scalar == 0
        single_pad = pad_size // scalar

        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes, num_tgt, single_pad, scalar

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

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels':
            self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            'masks':
            self.loss_masks,
            'boxes':
            self.loss_boxes_panoptic if self.panoptic_on else self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, mask_dict=None):
        """This performs the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.dn is not 'no' and mask_dict is not None:
            output_known_lbs_bboxes, num_tgt, single_pad, scalar = self.prep_for_dn(
                mask_dict)
            exc_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0,
                                     len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) *
                                  single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                exc_idx.append((output_idx, tgt_idx))
        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor([num_masks],
                                    dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_masks))

        if self.dn != 'no' and mask_dict is not None:
            l_dict = {}
            for loss in self.dn_losses:
                l_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets,
                                  exc_idx, num_masks * scalar))
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        elif self.dn != 'no':
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            if self.dn == 'seg':
                l_dict['loss_mask_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_dice_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_masks)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if 'interm_outputs' in outputs:
                    start = 0
                else:
                    start = 1
                if i >= start:
                    if self.dn != 'no' and mask_dict is not None:
                        out_ = output_known_lbs_bboxes['aux_outputs'][i]
                        l_dict = {}
                        for loss in self.dn_losses:
                            l_dict.update(
                                self.get_loss(loss, out_, targets, exc_idx,
                                              num_masks * scalar))
                        l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    elif self.dn != 'no':
                        l_dict = dict()
                        l_dict[f'loss_bbox_dn_{i}'] = torch.as_tensor(0.).to(
                            'cuda')
                        l_dict[f'loss_giou_dn_{i}'] = torch.as_tensor(0.).to(
                            'cuda')
                        l_dict[f'loss_ce_dn_{i}'] = torch.as_tensor(0.).to(
                            'cuda')
                        if self.dn == 'seg':
                            l_dict[f'loss_mask_dn_{i}'] = torch.as_tensor(
                                0.).to('cuda')
                            l_dict[f'loss_dice_dn_{i}'] = torch.as_tensor(
                                0.).to('cuda')
                        losses.update(l_dict)
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, interm_outputs, targets, indices,
                                       num_masks)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def __repr__(self):
        head = 'Criterion ' + self.__class__.__name__
        body = [
            'matcher: {}'.format(self.matcher.__repr__(_repr_indent=8)),
            'losses: {}'.format(self.losses),
            'weight_dict: {}'.format(self.weight_dict),
            'num_classes: {}'.format(self.num_classes),
            'eos_coef: {}'.format(self.eos_coef),
            'num_points: {}'.format(self.num_points),
            'oversample_ratio: {}'.format(self.oversample_ratio),
            'importance_sample_ratio: {}'.format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [' ' * _repr_indent + line for line in body]
        return '\n'.join(lines)


def get_uncertain_point_coords_with_randomness(coarse_logits, uncertainty_func,
                                               num_points, oversample_ratio,
                                               importance_sample_ratio):
    """Sample points in [0, 1] x [0, 1] coordinate space based on their
    uncertainty. The unceratinties are calculated for each point using
    'uncertainty_func' function that takes point's logit prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(
        num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(
        coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(
        point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(
                    num_boxes,
                    num_random_points,
                    2,
                    device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """Efficient version of torch.cat that avoids a copy if there is only a
    single element in a list."""
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def _process_criterion_info(class_weight,
                            mask_weight,
                            dice_weight,
                            box_weight,
                            giou_weight,
                            dn,
                            dec_layers,
                            box_loss,
                            two_stage=True,
                            deep_supervision=True):
    weight_dict = {'loss_ce': class_weight}
    weight_dict.update({'loss_mask': mask_weight, 'loss_dice': dice_weight})
    weight_dict.update({'loss_bbox': box_weight, 'loss_giou': giou_weight})
    # two stage is the query selection scheme
    if two_stage:
        interm_weight_dict = {}
        interm_weight_dict.update(
            {k + f'_interm': v
             for k, v in weight_dict.items()})
        weight_dict.update(interm_weight_dict)
    # denoising training
    if dn == 'standard':
        weight_dict.update({
            k + f'_dn': v
            for k, v in weight_dict.items()
            if k != 'loss_mask' and k != 'loss_dice'
        })
        dn_losses = ['labels', 'boxes']
    elif dn == 'seg':
        weight_dict.update({k + f'_dn': v for k, v in weight_dict.items()})
        dn_losses = ['labels', 'masks', 'boxes']
    else:
        dn_losses = []
    if deep_supervision:
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update(
                {k + f'_{i}': v
                 for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    if box_loss:
        losses = ['labels', 'masks', 'boxes']
    else:
        losses = ['labels', 'masks']
    return weight_dict, losses, dn_losses
