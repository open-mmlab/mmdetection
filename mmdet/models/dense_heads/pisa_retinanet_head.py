# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.core import images_to_levels
from ..builder import HEADS
from ..losses import carl_loss, isr_p
from .retina_head import RetinaHead


@HEADS.register_module()
class PISARetinaHead(RetinaHead):
    """PISA Retinanet Head.

    The head owns the same structure with Retinanet Head, but differs in two
        aspects:
        1. Importance-based Sample Reweighting Positive (ISR-P) is applied to
            change the positive loss weights.
        2. Classification-aware regression loss is adopted as a third loss.
    """

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image
                with shape (num_obj, 4).
            gt_labels (list[Tensor]): Ground truth labels of each image
                with shape (num_obj, 4).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor]): Ignored gt bboxes of each image.
                Default: None.

        Returns:
            dict: Loss dict, comprise classification loss, regression loss and
                carl loss.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            return_sampling_results=True)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, sampling_results_list) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        num_imgs = len(img_metas)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, label_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores = torch.cat(
            flatten_cls_scores, dim=1).reshape(-1,
                                               flatten_cls_scores[0].size(-1))
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds = torch.cat(
            flatten_bbox_preds, dim=1).view(-1, flatten_bbox_preds[0].size(-1))
        flatten_labels = torch.cat(labels_list, dim=1).reshape(-1)
        flatten_label_weights = torch.cat(
            label_weights_list, dim=1).reshape(-1)
        flatten_anchors = torch.cat(all_anchor_list, dim=1).reshape(-1, 4)
        flatten_bbox_targets = torch.cat(
            bbox_targets_list, dim=1).reshape(-1, 4)
        flatten_bbox_weights = torch.cat(
            bbox_weights_list, dim=1).reshape(-1, 4)

        # Apply ISR-P
        isr_cfg = self.train_cfg.get('isr', None)
        if isr_cfg is not None:
            all_targets = (flatten_labels, flatten_label_weights,
                           flatten_bbox_targets, flatten_bbox_weights)
            with torch.no_grad():
                all_targets = isr_p(
                    flatten_cls_scores,
                    flatten_bbox_preds,
                    all_targets,
                    flatten_anchors,
                    sampling_results_list,
                    bbox_coder=self.bbox_coder,
                    loss_cls=self.loss_cls,
                    num_class=self.num_classes,
                    **self.train_cfg.isr)
            (flatten_labels, flatten_label_weights, flatten_bbox_targets,
             flatten_bbox_weights) = all_targets

        # For convenience we compute loss once instead separating by fpn level,
        # so that we don't need to separate the weights by level again.
        # The result should be the same
        losses_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_labels,
            flatten_label_weights,
            avg_factor=num_total_samples)
        losses_bbox = self.loss_bbox(
            flatten_bbox_preds,
            flatten_bbox_targets,
            flatten_bbox_weights,
            avg_factor=num_total_samples)
        loss_dict = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

        # CARL Loss
        carl_cfg = self.train_cfg.get('carl', None)
        if carl_cfg is not None:
            loss_carl = carl_loss(
                flatten_cls_scores,
                flatten_labels,
                flatten_bbox_preds,
                flatten_bbox_targets,
                self.loss_bbox,
                **self.train_cfg.carl,
                avg_factor=num_total_pos,
                sigmoid=True,
                num_class=self.num_classes)
            loss_dict.update(loss_carl)

        return loss_dict
