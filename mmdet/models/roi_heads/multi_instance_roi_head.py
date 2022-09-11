# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..utils import unpack_gt_instances
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class MultiInstanceRoIHead(StandardRoIHead):

    def __init__(self, num_instance: int = 2, *args, **kwargs) -> None:
        self.num_instance = num_instance
        super().__init__(*args, **kwargs)

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
                       bbox_head: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        bbox_head.update(num_instance=self.num_instance)
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        num_imgs = len(batch_data_samples)
        top_k = self.num_instance

        # get targets
        return_rois = []
        return_labels = []
        return_bbox_targets = []
        for i in range(num_imgs):
            gt_boxes = batch_gt_instances[i].bboxes
            all_rois = torch.cat([rpn_results_list[i].bboxes, gt_boxes], dim=0)
            overlaps_normal, overlaps_ignore = box_overlap_ignore_opr(
                all_rois,
                torch.cat(
                    [gt_boxes, batch_gt_instances[i].labels.reshape(-1, 1)],
                    dim=1))
            overlaps_normal, overlaps_normal_indices = overlaps_normal.sort(
                descending=True, dim=1)
            overlaps_ignore, overlaps_ignore_indices = overlaps_ignore.sort(
                descending=True, dim=1)

            # gt max and indices, ignore max and indices
            max_overlaps_normal = overlaps_normal[:, :top_k].flatten()
            gt_assignment_normal = overlaps_normal_indices[:, :top_k].flatten()
            max_overlaps_ignore = overlaps_ignore[:, :top_k].flatten()
            gt_assignment_ignore = overlaps_ignore_indices[:, :top_k].flatten()

            # cons masks
            fg_threshold = self.train_cfg['assigner']['pos_iou_thr']
            ignore_label = self.train_cfg['assigner']['ignore_iof_thr']
            bg_threshold_high = self.train_cfg['assigner']['neg_iou_thr']

            ignore_assign_mask = (max_overlaps_normal < fg_threshold) * (
                max_overlaps_ignore > max_overlaps_normal)
            max_overlaps = (max_overlaps_normal * ~ignore_assign_mask) + (
                max_overlaps_ignore * ignore_assign_mask)
            gt_assignment = (gt_assignment_normal * ~ignore_assign_mask) + (
                gt_assignment_ignore * ignore_assign_mask)

            labels = batch_gt_instances[i].labels[gt_assignment]
            fg_mask = (max_overlaps >= fg_threshold) * (labels != ignore_label)
            bg_mask = (max_overlaps < bg_threshold_high) * (max_overlaps >= 0)
            fg_mask = fg_mask.reshape(-1, top_k)
            bg_mask = bg_mask.reshape(-1, top_k)

            pos_max = self.train_cfg['sampler']['num'] * self.train_cfg[
                'sampler']['pos_fraction']
            fg_inds_mask = subsample_masks(fg_mask[:, 0], pos_max, True)
            neg_max = self.train_cfg['sampler']['num'] - fg_inds_mask.sum()
            bg_inds_mask = subsample_masks(bg_mask[:, 0], neg_max, True)
            labels = labels + ~fg_mask.flatten(
            )  # labels = labels * fg_mask.flatten()
            keep_mask = fg_inds_mask + bg_inds_mask
            # labels
            labels = labels.reshape(-1, top_k)[keep_mask]
            gt_assignment = gt_assignment.reshape(-1,
                                                  top_k)[keep_mask].flatten()
            target_boxes = gt_boxes[gt_assignment, :4]
            rois = all_rois[keep_mask]
            target_rois = rois.repeat(1, top_k).reshape(-1, all_rois.shape[-1])
            bbox_targets = self.bbox_head.bbox_coder.encode(
                target_rois, target_boxes)
            bbox_targets = bbox_targets.reshape(-1, top_k * 4)
            return_rois.append(rois)
            return_labels.append(labels)
            return_bbox_targets.append(bbox_targets)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            rois = bbox2roi(return_rois)
            bbox_results = self._bbox_forward(x, rois)

            if len(bbox_results) == 2:
                loss0 = self.bbox_head.emd_loss_softmax(
                    bbox_results[1][:, :4], bbox_results[0][:, :2],
                    bbox_results[1][:, 4:], bbox_results[0][:, 2:],
                    torch.cat(return_bbox_targets, dim=0),
                    torch.cat(return_labels, dim=0))
                loss1 = self.bbox_head.emd_loss_softmax(
                    bbox_results[1][:, 4:], bbox_results[0][:, 2:],
                    bbox_results[1][:, :4], bbox_results[0][:, :2],
                    torch.cat(return_bbox_targets, dim=0),
                    torch.cat(return_labels, dim=0))
                loss = torch.cat([loss0, loss1], dim=1)
                _, min_indices = loss.min(dim=1)
                loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
                loss_emd = loss_emd.mean()
                losses['loss_rcnn_emd'] = loss_emd
            else:
                # TODO: add refine model
                pass

        return losses

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        bbox_results = self.bbox_head(bbox_feats)
        return bbox_results


def box_overlap_ignore_opr(box, gt, ignore_label=-1):
    assert box.ndim == 2
    assert gt.ndim == 2
    assert gt.shape[-1] > 4
    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    width_height = torch.min(box[:, None, 2:], gt[:, 2:4]) - torch.max(
        box[:, None, :2], gt[:, :2])  # [N,M,2]
    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height
    # handle empty boxes
    iou = torch.where(inter > 0, inter / (area_box[:, None] + area_gt - inter),
                      torch.zeros(1, dtype=inter.dtype, device=inter.device))
    ioa = torch.where(inter > 0, inter / (area_box[:, None]),
                      torch.zeros(1, dtype=inter.dtype, device=inter.device))
    gt_ignore_mask = gt[:, 4].eq(ignore_label).repeat(box.shape[0], 1)
    iou *= ~gt_ignore_mask
    ioa *= gt_ignore_mask
    return iou, ioa


def subsample_masks(masks, num_samples, sample_value):
    positive = torch.nonzero(masks.eq(sample_value), as_tuple=False).squeeze(1)
    num_mask = len(positive)
    num_samples = int(num_samples)
    num_final_samples = min(num_mask, num_samples)
    num_final_negative = num_mask - num_final_samples
    perm = torch.arange(num_mask, )[:num_final_negative]  # changed by yu
    # perm = torch.randperm(num_mask, device=masks.device)[:num_final_negative]
    negative = positive[perm]
    masks[negative] = not sample_value
    return masks
