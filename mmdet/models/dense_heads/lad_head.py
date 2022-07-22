# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import InstanceList, OptInstanceList
from ..utils import levels_to_images, multi_apply, unpack_gt_instances
from .paa_head import PAAHead


@MODELS.register_module()
class LADHead(PAAHead):
    """Label Assignment Head from the paper: `Improving Object Detection by
    Label Assignment Distillation <https://arxiv.org/pdf/2108.10520.pdf>`_"""

    def get_label_assignment(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            iou_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> tuple:
        """Get label assignment (from teacher).

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            iou_preds (list[Tensor]): iou_preds for each scale
                level with shape (N, num_anchors * 1, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            tuple: Returns a tuple containing label assignment variables.

            - labels (Tensor): Labels of all anchors, each with
              shape (num_anchors,).
            - labels_weight (Tensor): Label weights of all anchor.
              each with shape (num_anchors,).
            - bboxes_target (Tensor): BBox targets of all anchors.
              each with shape (num_anchors, 4).
            - bboxes_weight (Tensor): BBox weights of all anchors.
              each with shape (num_anchors, 4).
            - pos_inds_flatten (Tensor): Contains all index of positive
              sample in all anchor.
            - pos_anchors (Tensor): Positive anchors.
            - num_pos (int): Number of positive anchors.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )
        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds,
         pos_gt_index) = cls_reg_targets
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list,
                                       cls_scores, bbox_preds, labels,
                                       labels_weight, bboxes_target,
                                       bboxes_weight, pos_inds)

        with torch.no_grad():
            reassign_labels, reassign_label_weight, \
                reassign_bbox_weights, num_pos = multi_apply(
                    self.paa_reassign,
                    pos_losses_list,
                    labels,
                    labels_weight,
                    bboxes_weight,
                    pos_inds,
                    pos_gt_index,
                    anchor_list)
            num_pos = sum(num_pos)
        # convert all tensor list to a flatten tensor
        labels = torch.cat(reassign_labels, 0).view(-1)
        flatten_anchors = torch.cat(
            [torch.cat(item, 0) for item in anchor_list])
        labels_weight = torch.cat(reassign_label_weight, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target,
                                  0).view(-1, bboxes_target[0].size(-1))

        pos_inds_flatten = ((labels >= 0)
                            &
                            (labels < self.num_classes)).nonzero().reshape(-1)

        if num_pos:
            pos_anchors = flatten_anchors[pos_inds_flatten]
        else:
            pos_anchors = None

        label_assignment_results = (labels, labels_weight, bboxes_target,
                                    bboxes_weight, pos_inds_flatten,
                                    pos_anchors, num_pos)
        return label_assignment_results

    def loss(self, x: List[Tensor], label_assignment_results: tuple,
             batch_data_samples: SampleList) -> dict:
        """Forward train with the available label assignment (student receives
        from teacher).

        Args:
            x (list[Tensor]): Features from FPN.
            label_assignment_results (tuple): As the outputs defined in the
                function `self.get_label_assignment`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        outs = self(x)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(
            *loss_inputs,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            label_assignment_results=label_assignment_results)
        return losses

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     iou_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     label_assignment_results: Optional[tuple] = None) -> dict:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            iou_preds (list[Tensor]): iou_preds for each scale
                level with shape (N, num_anchors * 1, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            label_assignment_results (tuple, optional): As the outputs defined
                in the function `self.get_
                label_assignment`.

        Returns:
            dict[str, Tensor]: A dictionary of loss gmm_assignment.
        """

        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds_flatten,
         pos_anchors, num_pos) = label_assignment_results

        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]

        # convert all tensor list to a flatten tensor
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))

        losses_cls = self.loss_cls(
            cls_scores,
            labels,
            labels_weight,
            avg_factor=max(num_pos, len(batch_img_metas)))  # avoid num_pos=0
        if num_pos:
            pos_bbox_pred = self.bbox_coder.decode(
                pos_anchors, bbox_preds[pos_inds_flatten])
            pos_bbox_target = bboxes_target[pos_inds_flatten]
            iou_target = bbox_overlaps(
                pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)
            losses_iou = self.loss_centerness(
                iou_preds[pos_inds_flatten],
                iou_target.unsqueeze(-1),
                avg_factor=num_pos)
            losses_bbox = self.loss_bbox(
                pos_bbox_pred, pos_bbox_target, avg_factor=num_pos)

        else:
            losses_iou = iou_preds.sum() * 0
            losses_bbox = bbox_preds.sum() * 0

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_iou=losses_iou)
