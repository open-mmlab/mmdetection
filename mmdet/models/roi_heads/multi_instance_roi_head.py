# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmengine.structures.instance_data import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi, bbox_overlaps
from mmdet.utils import ConfigType, InstanceList
from ..utils import empty_instances, unpack_gt_instances
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

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:

        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # Set the FG label to 1 and add ignored annotations
        for i in range(len(batch_gt_instances)):
            for j in range(len(batch_gt_instances[i])):
                if batch_gt_instances[i].labels[j] == 0:
                    batch_gt_instances[i].labels[j] = 1

            batch_gt_instances[i] = InstanceData.cat(
                [batch_gt_instances[i], batch_gt_instances_ignore[i]])

        num_imgs = len(batch_data_samples)
        top_k = self.num_instance

        # get targets by meg
        rois = []
        labels = []
        bbox_targets = []
        for i in range(num_imgs):
            gt_boxes = batch_gt_instances[i].bboxes
            all_rois = torch.cat([rpn_results_list[i].bboxes, gt_boxes], dim=0)

            # calculate the iou of FG bboxes and ignore bboxes
            overlaps_normal = bbox_overlaps(all_rois, gt_boxes, mode='iou')
            overlaps_ignore = bbox_overlaps(all_rois, gt_boxes, mode='iof')
            gt_ignore_mask = batch_gt_instances[i].labels.eq(-1).repeat(
                all_rois.shape[0], 1)
            overlaps_normal = overlaps_normal * ~gt_ignore_mask
            overlaps_ignore = overlaps_ignore * gt_ignore_mask
            overlaps_normal, overlaps_normal_indices = overlaps_normal.sort(
                descending=True, dim=1)
            overlaps_ignore, overlaps_ignore_indices = overlaps_ignore.sort(
                descending=True, dim=1)

            # select the roi with the higher score
            max_overlaps_normal = overlaps_normal[:, :top_k].flatten()
            gt_assignment_normal = overlaps_normal_indices[:, :top_k].flatten()
            max_overlaps_ignore = overlaps_ignore[:, :top_k].flatten()
            gt_assignment_ignore = overlaps_ignore_indices[:, :top_k].flatten()

            # hyperparameters
            fg_threshold = self.train_cfg['assigner']['pos_iou_thr']
            ignore_label = self.train_cfg['assigner']['ignore_iof_thr']
            bg_threshold_high = self.train_cfg['assigner']['neg_iou_thr']

            # ignore or not
            ignore_assign_mask = (max_overlaps_normal < fg_threshold) * (
                max_overlaps_ignore > max_overlaps_normal)
            max_overlaps = (max_overlaps_normal * ~ignore_assign_mask) + (
                max_overlaps_ignore * ignore_assign_mask)
            gt_assignment = (gt_assignment_normal * ~ignore_assign_mask) + (
                gt_assignment_ignore * ignore_assign_mask)

            # assign positive and negative samples
            _labels = batch_gt_instances[i].labels[gt_assignment]
            fg_mask = (max_overlaps >= fg_threshold) * (
                _labels != ignore_label)
            bg_mask = (max_overlaps < bg_threshold_high) * (max_overlaps >= 0)
            fg_mask = fg_mask.reshape(-1, top_k)
            bg_mask = bg_mask.reshape(-1, top_k)

            # sampler
            pos_max = self.train_cfg['sampler']['num'] * self.train_cfg[
                'sampler']['pos_fraction']
            fg_inds_mask = subsample_masks(fg_mask[:, 0], pos_max, True)
            neg_max = self.train_cfg['sampler']['num'] - fg_inds_mask.sum()
            bg_inds_mask = subsample_masks(bg_mask[:, 0], neg_max, True)
            _labels = _labels * fg_mask.flatten()
            keep_mask = fg_inds_mask + bg_inds_mask
            _labels = _labels.reshape(-1, top_k)[keep_mask]
            gt_assignment = gt_assignment.reshape(-1,
                                                  top_k)[keep_mask].flatten()
            target_boxes = gt_boxes[gt_assignment, :4]
            _rois = all_rois[keep_mask]
            target_rois = _rois.repeat(1,
                                       top_k).reshape(-1, all_rois.shape[-1])

            # bboxes encode
            _bbox_targets = self.bbox_head.bbox_coder.encode(
                target_rois, target_boxes)
            _bbox_targets = _bbox_targets.reshape(-1, top_k * 4)

            rois.append(_rois)
            labels.append(_labels)
            bbox_targets.append(_bbox_targets)

        return_labels = torch.cat(labels, dim=0)
        return_bbox_targets = torch.cat(bbox_targets, dim=0)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            rois = bbox2roi(rois)
            bbox_results = self._bbox_forward(x, rois)

            if len(bbox_results) == 2:
                loss0 = self.bbox_head.emd_loss_softmax(
                    bbox_results[1][:, :4], bbox_results[0][:, :2],
                    bbox_results[1][:, 4:], bbox_results[0][:, 2:],
                    return_bbox_targets, return_labels)
                loss1 = self.bbox_head.emd_loss_softmax(
                    bbox_results[1][:, 4:], bbox_results[0][:, 2:],
                    bbox_results[1][:, :4], bbox_results[0][:, :2],
                    return_bbox_targets, return_labels)
                loss = torch.cat([loss0, loss1], dim=1)
                _, min_indices = loss.min(dim=1)
                loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
                loss_emd = loss_emd.mean()
                losses['loss_rcnn_emd'] = loss_emd
            else:
                # TODO: add refine model
                pass

        return losses

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas, rois.device, task_type='bbox')

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        if len(bbox_results) == 2:
            cls_scores = bbox_results[0]
            bbox_preds = bbox_results[1]
            num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_scores = cls_scores.split(num_proposals_per_img, 0)

            # some detector with_reg is False, bbox_preds will be None
            if bbox_preds is not None:
                # TODO move this to a sabl_roi_head
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_preds, torch.Tensor):
                    bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
                else:
                    bbox_preds = self.bbox_head.bbox_pred_split(
                        bbox_preds, num_proposals_per_img)
            else:
                bbox_preds = (None, ) * len(proposals)

            result_list = self.bbox_head.predict_by_feat(
                rois=rois,
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                batch_img_metas=batch_img_metas,
                rcnn_test_cfg=rcnn_test_cfg,
                rescale=rescale)
            return result_list
        else:
            # TODO: refine model.
            return proposals


def subsample_masks(masks, num_samples, sample_value):
    positive = torch.nonzero(masks.eq(sample_value), as_tuple=False).squeeze(1)
    num_mask = len(positive)
    num_samples = int(num_samples)
    num_final_samples = min(num_mask, num_samples)
    num_final_negative = num_mask - num_final_samples
    # perm = torch.arange(num_mask, )[:num_final_negative]  # changed by yu
    perm = torch.randperm(num_mask, device=masks.device)[:num_final_negative]
    negative = positive[perm]
    masks[negative] = not sample_value
    return masks


def bbox_transform_opr(bbox, gt):
    """Transform the bounding box and ground truth to the loss targets.

    The 4 box coordinates are in axis 1
    """
    bbox_width = bbox[:, 2] - bbox[:, 0] + 1
    bbox_height = bbox[:, 3] - bbox[:, 1] + 1
    bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height

    gt_width = gt[:, 2] - gt[:, 0] + 1
    gt_height = gt[:, 3] - gt[:, 1] + 1
    gt_ctr_x = gt[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt[:, 1] + 0.5 * gt_height

    target_dx = (gt_ctr_x - bbox_ctr_x) / bbox_width
    target_dy = (gt_ctr_y - bbox_ctr_y) / bbox_height
    target_dw = torch.log(gt_width / bbox_width)
    target_dh = torch.log(gt_height / bbox_height)
    target = torch.cat((target_dx.reshape(-1, 1), target_dy.reshape(
        -1, 1), target_dw.reshape(-1, 1), target_dh.reshape(-1, 1)),
                       dim=1)
    return target
