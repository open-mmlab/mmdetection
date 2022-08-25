# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import InstanceList, OptConfigType, OptInstanceList
from ..utils import multi_apply
from .retina_head import RetinaHead

EPS = 1e-12


@MODELS.register_module()
class FreeAnchorRetinaHead(RetinaHead):
    """FreeAnchor RetinaHead used in https://arxiv.org/abs/1909.02466.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Defaults to 4.
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to
            construct and config conv layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): dictionary to
            construct and config norm layer. Defaults to
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        pre_anchor_topk (int): Number of boxes that be token in each bag.
            Defaults to 50
        bbox_thr (float): The threshold of the saturated linear function.
            It is usually the same with the IoU threshold used in NMS.
            Defaults to 0.6.
        gamma (float): Gamma parameter in focal loss. Defaults to 2.0.
        alpha (float): Alpha parameter in focal loss. Defaults to 0.5.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 pre_anchor_topk: int = 50,
                 bbox_thr: float = 0.6,
                 gamma: float = 2.0,
                 alpha: float = 0.5,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            **kwargs)

        self.pre_anchor_topk = pre_anchor_topk
        self.bbox_thr = bbox_thr
        self.gamma = gamma
        self.alpha = alpha

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, _ = self.get_anchors(
            featmap_sizes=featmap_sizes,
            batch_img_metas=batch_img_metas,
            device=device)
        concat_anchor_list = [torch.cat(anchor) for anchor in anchor_list]

        # concatenate each level
        cls_scores = [
            cls.permute(0, 2, 3,
                        1).reshape(cls.size(0), -1, self.cls_out_channels)
            for cls in cls_scores
        ]
        bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.size(0), -1, 4)
            for bbox_pred in bbox_preds
        ]
        cls_scores = torch.cat(cls_scores, dim=1)
        cls_probs = torch.sigmoid(cls_scores)
        bbox_preds = torch.cat(bbox_preds, dim=1)

        box_probs, positive_losses, num_pos_list = multi_apply(
            self.positive_loss_single, cls_probs, bbox_preds,
            concat_anchor_list, batch_gt_instances)

        num_pos = sum(num_pos_list)
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

        # box_prob: P{a_{j} \in A_{+}}
        box_probs = torch.stack(box_probs, dim=0)

        # negative_loss:
        # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
        negative_loss = self.negative_bag_loss(cls_probs, box_probs).sum() / \
            max(1, num_pos * self.pre_anchor_topk)

        # avoid the absence of gradients in regression subnet
        # when no ground-truth in a batch
        if num_pos == 0:
            positive_loss = bbox_preds.sum() * 0

        losses = {
            'positive_bag_loss': positive_loss,
            'negative_bag_loss': negative_loss
        }
        return losses

    def positive_loss_single(self, cls_prob: Tensor, bbox_pred: Tensor,
                             flat_anchors: Tensor,
                             gt_instances: InstanceData) -> tuple:
        """Compute positive loss.

        Args:
            cls_prob (Tensor): Classification probability of shape
                (num_anchors, num_classes).
            bbox_pred (Tensor): Box probability of shape (num_anchors, 4).
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple:

                - box_prob (Tensor): Box probability of shape (num_anchors, 4).
                - positive_loss (Tensor): Positive loss of shape (num_pos, ).
                - num_pos (int): positive samples indexes.
        """

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        with torch.no_grad():
            if len(gt_bboxes) == 0:
                image_box_prob = torch.zeros(
                    flat_anchors.size(0),
                    self.cls_out_channels).type_as(bbox_pred)
            else:
                # box_localization: a_{j}^{loc}, shape: [j, 4]
                pred_boxes = self.bbox_coder.decode(flat_anchors, bbox_pred)

                # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                object_box_iou = bbox_overlaps(gt_bboxes, pred_boxes)

                # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                t1 = self.bbox_thr
                t2 = object_box_iou.max(
                    dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(
                    min=0, max=1)

                # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                num_obj = gt_labels.size(0)
                indices = torch.stack(
                    [torch.arange(num_obj).type_as(gt_labels), gt_labels],
                    dim=0)
                object_cls_box_prob = torch.sparse_coo_tensor(
                    indices, object_box_prob)

                # image_box_iou: P{a_{j} \in A_{+}}, shape: [c, j]
                """
                from "start" to "end" implement:
                image_box_iou = torch.sparse.max(object_cls_box_prob,
                                                 dim=0).t()

                """
                # start
                box_cls_prob = torch.sparse.sum(
                    object_cls_box_prob, dim=0).to_dense()

                indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()
                if indices.numel() == 0:
                    image_box_prob = torch.zeros(
                        flat_anchors.size(0),
                        self.cls_out_channels).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where(
                        (gt_labels.unsqueeze(dim=-1) == indices[0]),
                        object_box_prob[:, indices[1]],
                        torch.tensor(
                            [0]).type_as(object_box_prob)).max(dim=0).values

                    # upmap to shape [j, c]
                    image_box_prob = torch.sparse_coo_tensor(
                        indices.flip([0]),
                        nonzero_box_prob,
                        size=(flat_anchors.size(0),
                              self.cls_out_channels)).to_dense()
                # end
            box_prob = image_box_prob

        # construct bags for objects
        match_quality_matrix = bbox_overlaps(gt_bboxes, flat_anchors)
        _, matched = torch.topk(
            match_quality_matrix, self.pre_anchor_topk, dim=1, sorted=False)
        del match_quality_matrix

        # matched_cls_prob: P_{ij}^{cls}
        matched_cls_prob = torch.gather(
            cls_prob[matched], 2,
            gt_labels.view(-1, 1, 1).repeat(1, self.pre_anchor_topk,
                                            1)).squeeze(2)

        # matched_box_prob: P_{ij}^{loc}
        matched_anchors = flat_anchors[matched]
        matched_object_targets = self.bbox_coder.encode(
            matched_anchors,
            gt_bboxes.unsqueeze(dim=1).expand_as(matched_anchors))
        loss_bbox = self.loss_bbox(
            bbox_pred[matched],
            matched_object_targets,
            reduction_override='none').sum(-1)
        matched_box_prob = torch.exp(-loss_bbox)

        # positive_losses: {-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )}
        num_pos = len(gt_bboxes)
        positive_loss = self.positive_bag_loss(matched_cls_prob,
                                               matched_box_prob)

        return box_prob, positive_loss, num_pos

    def positive_bag_loss(self, matched_cls_prob: Tensor,
                          matched_box_prob: Tensor) -> Tensor:
        """Compute positive bag loss.

        :math:`-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )`.

        :math:`P_{ij}^{cls}`: matched_cls_prob, classification probability of matched samples.

        :math:`P_{ij}^{loc}`: matched_box_prob, box probability of matched samples.

        Args:
            matched_cls_prob (Tensor): Classification probability of matched
                samples in shape (num_gt, pre_anchor_topk).
            matched_box_prob (Tensor): BBox probability of matched samples,
                in shape (num_gt, pre_anchor_topk).

        Returns:
            Tensor: Positive bag loss in shape (num_gt,).
        """  # noqa: E501, W605
        # bag_prob = Mean-max(matched_prob)
        matched_prob = matched_cls_prob * matched_box_prob
        weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
        weight /= weight.sum(dim=1).unsqueeze(dim=-1)
        bag_prob = (weight * matched_prob).sum(dim=1)
        # positive_bag_loss = -self.alpha * log(bag_prob)
        return self.alpha * F.binary_cross_entropy(
            bag_prob, torch.ones_like(bag_prob), reduction='none')

    def negative_bag_loss(self, cls_prob: Tensor, box_prob: Tensor) -> Tensor:
        """Compute negative bag loss.

        :math:`FL((1 - P_{a_{j} \in A_{+}}) * (1 - P_{j}^{bg}))`.

        :math:`P_{a_{j} \in A_{+}}`: Box_probability of matched samples.

        :math:`P_{j}^{bg}`: Classification probability of negative samples.

        Args:
            cls_prob (Tensor): Classification probability, in shape
                (num_img, num_anchors, num_classes).
            box_prob (Tensor): Box probability, in shape
                (num_img, num_anchors, num_classes).

        Returns:
            Tensor: Negative bag loss in shape (num_img, num_anchors,
            num_classes).
        """  # noqa: E501, W605
        prob = cls_prob * (1 - box_prob)
        # There are some cases when neg_prob = 0.
        # This will cause the neg_prob.log() to be inf without clamp.
        prob = prob.clamp(min=EPS, max=1 - EPS)
        negative_bag_loss = prob**self.gamma * F.binary_cross_entropy(
            prob, torch.zeros_like(prob), reduction='none')
        return (1 - self.alpha) * negative_bag_loss
