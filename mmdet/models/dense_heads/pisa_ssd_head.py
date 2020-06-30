import torch

from mmdet.core import multi_apply
from ..builder import HEADS
from ..losses import CrossEntropyLoss, SmoothL1Loss, carl_loss, isr_p
from .ssd_head import SSDHead


# TODO: add loss evaluator for SSD
@HEADS.register_module()
class PISASSDHead(SSDHead):

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
            dict: Loss dict, comprise classification loss regression loss and
                carl loss.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False,
            return_sampling_results=True)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, sampling_results_list) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        isr_cfg = self.train_cfg.get('isr', None)
        all_targets = (all_labels.view(-1), all_label_weights.view(-1),
                       all_bbox_targets.view(-1,
                                             4), all_bbox_weights.view(-1, 4))
        # apply ISR-P
        if isr_cfg is not None:
            all_targets = isr_p(
                all_cls_scores.view(-1, all_cls_scores.size(-1)),
                all_bbox_preds.view(-1, 4),
                all_targets,
                torch.cat(all_anchors),
                sampling_results_list,
                loss_cls=CrossEntropyLoss(),
                bbox_coder=self.bbox_coder,
                **self.train_cfg.isr,
                num_class=self.num_classes)
            (new_labels, new_label_weights, new_bbox_targets,
             new_bbox_weights) = all_targets
            all_labels = new_labels.view(all_labels.shape)
            all_label_weights = new_label_weights.view(all_label_weights.shape)
            all_bbox_targets = new_bbox_targets.view(all_bbox_targets.shape)
            all_bbox_weights = new_bbox_weights.view(all_bbox_weights.shape)

        # add CARL loss
        carl_loss_cfg = self.train_cfg.get('carl', None)
        if carl_loss_cfg is not None:
            loss_carl = carl_loss(
                all_cls_scores.view(-1, all_cls_scores.size(-1)),
                all_targets[0],
                all_bbox_preds.view(-1, 4),
                all_targets[2],
                SmoothL1Loss(beta=1.),
                **self.train_cfg.carl,
                avg_factor=num_total_pos,
                num_class=self.num_classes)

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        loss_dict = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        if carl_loss_cfg is not None:
            loss_dict.update(loss_carl)
        return loss_dict
