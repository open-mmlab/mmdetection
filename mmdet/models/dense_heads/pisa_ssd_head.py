# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptInstanceList
from ..losses import CrossEntropyLoss, SmoothL1Loss, carl_loss, isr_p
from ..utils import multi_apply
from .ssd_head import SSDHead


# TODO: add loss evaluator for SSD
@MODELS.register_module()
class PISASSDHead(SSDHead):
    """Implementation of `PISA SSD head <https://arxiv.org/abs/1904.04821>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Sequence[int]): Number of channels in the input feature
            map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Defaults to 0.
        feat_channels (int): Number of hidden channels when stacked_convs
            > 0. Defaults to 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, Optional): Dictionary to construct
            and config conv layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, Optional): Dictionary to construct
            and config norm layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, Optional): Dictionary to construct
            and config activation layer. Defaults to None.
        anchor_generator (:obj:`ConfigDict` or dict): Config dict for anchor
            generator.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Defaults to False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (:obj:`ConfigDict` or dict, Optional): Training config of
            anchor head.
        test_cfg (:obj:`ConfigDict` or dict, Optional): Testing config of
            anchor head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], Optional): Initialization config dict.
    """  # noqa: W605

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Union[List[Tensor], Tensor]]:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Union[List[Tensor], Tensor]]: A dictionary of loss
            components. the dict has components below:

            - loss_cls (list[Tensor]): A list containing each feature map \
            classification loss.
            - loss_bbox (list[Tensor]): A list containing each feature map \
            regression loss.
            - loss_carl (Tensor): The loss of CARL.
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
            unmap_outputs=False,
            return_sampling_results=True)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor, sampling_results_list) = cls_reg_targets

        num_images = len(batch_img_metas)
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
                **self.train_cfg['isr'],
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
                **self.train_cfg['carl'],
                avg_factor=avg_factor,
                num_class=self.num_classes)

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            avg_factor=avg_factor)
        loss_dict = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        if carl_loss_cfg is not None:
            loss_dict.update(loss_carl)
        return loss_dict
