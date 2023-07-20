# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/loss.py   # noqa

# This work is licensed under the CC-BY-NC 4.0 License.
# Users should be careful about adopting these features in any commercial matters.  # noqa
# For more details, please refer to https://github.com/ShoufaChen/DiffusionDet/blob/main/LICENSE    # noqa

from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType


@TASK_UTILS.register_module()
class DiffusionDetCriterion(nn.Module):

    def __init__(
            self,
            num_classes,
            assigner: Union[ConfigDict, nn.Module],
            deep_supervision=True,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=5.0),
            loss_giou=dict(type='GIoULoss', reduction='sum', loss_weight=2.0),
    ):

        super().__init__()
        self.num_classes = num_classes

        if isinstance(assigner, nn.Module):
            self.assigner = assigner
        else:
            self.assigner = TASK_UTILS.build(assigner)

        self.deep_supervision = deep_supervision

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_giou = MODELS.build(loss_giou)

    def forward(self, outputs, batch_gt_instances, batch_img_metas):
        batch_indices = self.assigner(outputs, batch_gt_instances,
                                      batch_img_metas)
        # Compute all the requested losses
        loss_cls = self.loss_classification(outputs, batch_gt_instances,
                                            batch_indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, batch_gt_instances,
                                               batch_indices)

        losses = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_giou=loss_giou)

        if self.deep_supervision:
            assert 'aux_outputs' in outputs
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                batch_indices = self.assigner(aux_outputs, batch_gt_instances,
                                              batch_img_metas)
                loss_cls = self.loss_classification(aux_outputs,
                                                    batch_gt_instances,
                                                    batch_indices)
                loss_bbox, loss_giou = self.loss_boxes(aux_outputs,
                                                       batch_gt_instances,
                                                       batch_indices)
                tmp_losses = dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_bbox,
                    loss_giou=loss_giou)
                for name, value in tmp_losses.items():
                    losses[f's.{i}.{name}'] = value
        return losses

    def loss_classification(self, outputs, batch_gt_instances, indices):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes_list = [
            gt.labels[J] for gt, (_, J) in zip(batch_gt_instances, indices)
        ]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        for idx in range(len(batch_gt_instances)):
            target_classes[idx, indices[idx][0]] = target_classes_list[idx]

        src_logits = src_logits.flatten(0, 1)
        target_classes = target_classes.flatten(0, 1)
        # comp focal loss.
        num_instances = max(torch.cat(target_classes_list).shape[0], 1)
        loss_cls = self.loss_cls(
            src_logits,
            target_classes,
        ) / num_instances
        return loss_cls

    def loss_boxes(self, outputs, batch_gt_instances, indices):
        assert 'pred_boxes' in outputs
        pred_boxes = outputs['pred_boxes']

        target_bboxes_norm_list = [
            gt.norm_bboxes_cxcywh[J]
            for gt, (_, J) in zip(batch_gt_instances, indices)
        ]
        target_bboxes_list = [
            gt.bboxes[J] for gt, (_, J) in zip(batch_gt_instances, indices)
        ]

        pred_bboxes_list = []
        pred_bboxes_norm_list = []
        for idx in range(len(batch_gt_instances)):
            pred_bboxes_list.append(pred_boxes[idx, indices[idx][0]])
            image_size = batch_gt_instances[idx].image_size
            pred_bboxes_norm_list.append(pred_boxes[idx, indices[idx][0]] /
                                         image_size)

        pred_boxes_cat = torch.cat(pred_bboxes_list)
        pred_boxes_norm_cat = torch.cat(pred_bboxes_norm_list)
        target_bboxes_cat = torch.cat(target_bboxes_list)
        target_bboxes_norm_cat = torch.cat(target_bboxes_norm_list)

        if len(pred_boxes_cat) > 0:
            num_instances = pred_boxes_cat.shape[0]

            loss_bbox = self.loss_bbox(
                pred_boxes_norm_cat,
                bbox_cxcywh_to_xyxy(target_bboxes_norm_cat)) / num_instances
            loss_giou = self.loss_giou(pred_boxes_cat,
                                       target_bboxes_cat) / num_instances
        else:
            loss_bbox = pred_boxes.sum() * 0
            loss_giou = pred_boxes.sum() * 0
        return loss_bbox, loss_giou


@TASK_UTILS.register_module()
class DiffusionDetMatcher(nn.Module):
    """This class computes an assignment between the targets and the
    predictions of the network For efficiency reasons, the targets don't
    include the no_object.

    Because of this, in general, there are more predictions than targets. In
    this case, we do a 1-to-k (dynamic) matching of the best predictions, while
    the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                    ConfigDict],
                 center_radius: float = 2.5,
                 candidate_topk: int = 5,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 **kwargs):
        super().__init__()

        self.center_radius = center_radius
        self.candidate_topk = candidate_topk

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'
        self.use_focal_loss = False
        self.use_fed_loss = False
        for _match_cost in match_costs:
            if _match_cost.get('type') == 'FocalLossCost':
                self.use_focal_loss = True
            if _match_cost.get('type') == 'FedLoss':
                self.use_fed_loss = True
                raise NotImplementedError

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def forward(self, outputs, batch_gt_instances, batch_img_metas):
        assert 'pred_logits' in outputs and 'pred_boxes' in outputs

        pred_logits = outputs['pred_logits']
        pred_bboxes = outputs['pred_boxes']
        batch_size = len(batch_gt_instances)

        assert batch_size == pred_logits.shape[0] == pred_bboxes.shape[0]
        batch_indices = []
        for i in range(batch_size):
            pred_instances = InstanceData()
            pred_instances.bboxes = pred_bboxes[i, ...]
            pred_instances.scores = pred_logits[i, ...]
            gt_instances = batch_gt_instances[i]
            img_meta = batch_img_metas[i]
            indices = self.single_assigner(pred_instances, gt_instances,
                                           img_meta)
            batch_indices.append(indices)
        return batch_indices

    def single_assigner(self, pred_instances, gt_instances, img_meta):
        with torch.no_grad():
            gt_bboxes = gt_instances.bboxes
            pred_bboxes = pred_instances.bboxes
            num_gt = gt_bboxes.size(0)

            if num_gt == 0:  # empty object in key frame
                valid_mask = pred_bboxes.new_zeros((pred_bboxes.shape[0], ),
                                                   dtype=torch.bool)
                matched_gt_inds = pred_bboxes.new_zeros((gt_bboxes.shape[0], ),
                                                        dtype=torch.long)
                return valid_mask, matched_gt_inds

            valid_mask, is_in_boxes_and_center = \
                self.get_in_gt_and_in_center_info(
                    bbox_xyxy_to_cxcywh(pred_bboxes),
                    bbox_xyxy_to_cxcywh(gt_bboxes)
                )

            cost_list = []
            for match_cost in self.match_costs:
                cost = match_cost(
                    pred_instances=pred_instances,
                    gt_instances=gt_instances,
                    img_meta=img_meta)
                cost_list.append(cost)

            pairwise_ious = self.iou_calculator(pred_bboxes, gt_bboxes)

            cost_list.append((~is_in_boxes_and_center) * 100.0)
            cost_matrix = torch.stack(cost_list).sum(0)
            cost_matrix[~valid_mask] = cost_matrix[~valid_mask] + 10000.0

            fg_mask_inboxes, matched_gt_inds = \
                self.dynamic_k_matching(
                    cost_matrix, pairwise_ious, num_gt)
        return fg_mask_inboxes, matched_gt_inds

    def get_in_gt_and_in_center_info(
            self, pred_bboxes: Tensor,
            gt_bboxes: Tensor) -> Tuple[Tensor, Tensor]:
        """Get the information of which prior is in gt bboxes and gt center
        priors."""
        xy_target_gts = bbox_cxcywh_to_xyxy(gt_bboxes)  # (x1, y1, x2, y2)

        pred_bboxes_center_x = pred_bboxes[:, 0].unsqueeze(1)
        pred_bboxes_center_y = pred_bboxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = pred_bboxes_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = pred_bboxes_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = pred_bboxes_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = pred_bboxes_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() +
                        b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends
        # on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212    # noqa
        b_l = pred_bboxes_center_x > (
            gt_bboxes[:, 0] -
            (center_radius *
             (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = pred_bboxes_center_x < (
            gt_bboxes[:, 0] +
            (center_radius *
             (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = pred_bboxes_center_y > (
            gt_bboxes[:, 1] -
            (center_radius *
             (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = pred_bboxes_center_y < (
            gt_bboxes[:, 1] +
            (center_radius *
             (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() +
                          b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,
                           num_gt: int) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets."""
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            _, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        while (matching_matrix.sum(0) == 0).any():
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(
                matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[prior_match_gt_mask], dim=1)
                matching_matrix[prior_match_gt_mask] *= 0
                matching_matrix[prior_match_gt_mask, cost_argmin, ] = 1

        assert not (matching_matrix.sum(0) == 0).any()
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)

        return fg_mask_inboxes, matched_gt_inds
