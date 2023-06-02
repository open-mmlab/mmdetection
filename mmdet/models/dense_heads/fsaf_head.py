# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptInstanceList, OptMultiConfig
from ..losses.accuracy import accuracy
from ..losses.utils import weight_reduce_loss
from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import images_to_levels, multi_apply, unmap
from .retina_head import RetinaHead


@MODELS.register_module()
class FSAFHead(RetinaHead):
    """Anchor-free head used in `FSAF <https://arxiv.org/abs/1903.00621>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors (num_anchors is 1 for anchor-
    free methods)

    Args:
        *args: Same as its base class in :class:`RetinaHead`
        score_threshold (float, optional): The score_threshold to calculate
            positive recall. If given, prediction scores lower than this value
            is counted as incorrect prediction. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
        **kwargs: Same as its base class in :class:`RetinaHead`

    Example:
        >>> import torch
        >>> self = FSAFHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == self.num_classes
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 *args,
                 score_threshold: Optional[float] = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        # The positive bias in self.retina_reg conv is to prevent predicted \
        #  bbox with 0 area
        if init_cfg is None:
            init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=[
                    dict(
                        type='Normal',
                        name='retina_cls',
                        std=0.01,
                        bias_prob=0.01),
                    dict(
                        type='Normal', name='retina_reg', std=0.01, bias=0.25)
                ])
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        self.score_threshold = score_threshold

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward feature map of a single scale level.

        Args:
            x (Tensor): Feature map of a single scale level.

        Returns:
            tuple[Tensor, Tensor]:

            - cls_score (Tensor): Box scores for each scale level Has \
            shape (N, num_points * num_classes, H, W).
            - bbox_pred (Tensor): Box energies / deltas for each scale \
            level with shape (N, num_points * 4, H, W).
        """
        cls_score, bbox_pred = super().forward_single(x)
        # relu: TBLR encoder only accepts positive bbox_pred
        return cls_score, self.relu(bbox_pred)

    def _get_targets_single(self,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in a
        single image.

        Most of the codes are the same with the base class :obj: `AnchorHead`,
        except that it also collects and returns the matched gt index in the
        image (from 0 to num_gt-1). If the anchor bbox is not matched to any
        gt, the corresponding value in pos_gt_inds is -1.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors, ).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.  Defaults to True.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # Assign gt and sample anchors
        anchors = flat_anchors[inside_flags.type(torch.bool), :]

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(
            (num_valid_anchors, self.cls_out_channels), dtype=torch.float)
        pos_gt_inds = anchors.new_full((num_valid_anchors, ),
                                       -1,
                                       dtype=torch.long)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            # The assigned gt_index for each anchor. (0-based)
            pos_gt_inds[pos_inds] = sampling_result.pos_assigned_gt_inds
            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # shadowed_labels is a tensor composed of tuples
        #  (anchor_inds, class_label) that indicate those anchors lying in the
        #  outer region of a gt or overlapped by another gt with a smaller
        #  area.
        #
        # Therefore, only the shadowed labels are ignored for loss calculation.
        # the key `shadowed_labels` is defined in :obj:`CenterRegionAssigner`
        shadowed_labels = assign_result.get_extra_property('shadowed_labels')
        if shadowed_labels is not None and shadowed_labels.numel():
            if len(shadowed_labels.shape) == 2:
                idx_, label_ = shadowed_labels[:, 0], shadowed_labels[:, 1]
                assert (labels[idx_] != label_).all(), \
                    'One label cannot be both positive and ignored'
                label_weights[idx_, label_] = 0
            else:
                label_weights[shadowed_labels] = 0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            pos_gt_inds = unmap(
                pos_gt_inds, num_total_anchors, inside_flags, fill=-1)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, pos_gt_inds)

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
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
            dict[str, Tensor]: A dictionary of loss components.
        """
        for i in range(len(bbox_preds)):  # loop over fpn level
            # avoid 0 area of the predicted bbox
            bbox_preds[i] = bbox_preds[i].clamp(min=1e-4)
        # TODO: It may directly use the base-class loss function.
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        batch_size = len(batch_img_metas)
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            return_sampling_results=True)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor, sampling_results_list,
         pos_assigned_gt_inds_list) = cls_reg_targets

        num_gts = np.array(list(map(len, batch_gt_instances)))
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor)

        # `pos_assigned_gt_inds_list` (length: fpn_levels) stores the assigned
        # gt index of each anchor bbox in each fpn level.
        cum_num_gts = list(np.cumsum(num_gts))  # length of batch_size
        for i, assign in enumerate(pos_assigned_gt_inds_list):
            # loop over fpn levels
            for j in range(1, batch_size):
                # loop over batch size
                # Convert gt indices in each img to those in the batch
                assign[j][assign[j] >= 0] += int(cum_num_gts[j - 1])
            pos_assigned_gt_inds_list[i] = assign.flatten()
            labels_list[i] = labels_list[i].flatten()
        num_gts = num_gts.sum()  # total number of gt in the batch
        # The unique label index of each gt in the batch
        label_sequence = torch.arange(num_gts, device=device)
        # Collect the average loss of each gt in each level
        with torch.no_grad():
            loss_levels, = multi_apply(
                self.collect_loss_level_single,
                losses_cls,
                losses_bbox,
                pos_assigned_gt_inds_list,
                labels_seq=label_sequence)
            # Shape: (fpn_levels, num_gts). Loss of each gt at each fpn level
            loss_levels = torch.stack(loss_levels, dim=0)
            # Locate the best fpn level for loss back-propagation
            if loss_levels.numel() == 0:  # zero gt
                argmin = loss_levels.new_empty((num_gts, ), dtype=torch.long)
            else:
                _, argmin = loss_levels.min(dim=0)

        # Reweight the loss of each (anchor, label) pair, so that only those
        #  at the best gt level are back-propagated.
        losses_cls, losses_bbox, pos_inds = multi_apply(
            self.reweight_loss_single,
            losses_cls,
            losses_bbox,
            pos_assigned_gt_inds_list,
            labels_list,
            list(range(len(losses_cls))),
            min_levels=argmin)
        num_pos = torch.cat(pos_inds, 0).sum().float()
        pos_recall = self.calculate_pos_recall(cls_scores, labels_list,
                                               pos_inds)

        if num_pos == 0:  # No gt
            num_total_neg = sum(
                [results.num_neg for results in sampling_results_list])
            avg_factor = num_pos + num_total_neg
        else:
            avg_factor = num_pos
        for i in range(len(losses_cls)):
            losses_cls[i] /= avg_factor
            losses_bbox[i] /= avg_factor
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            num_pos=num_pos / batch_size,
            pos_recall=pos_recall)

    def calculate_pos_recall(self, cls_scores: List[Tensor],
                             labels_list: List[Tensor],
                             pos_inds: List[Tensor]) -> Tensor:
        """Calculate positive recall with score threshold.

        Args:
            cls_scores (list[Tensor]): Classification scores at all fpn levels.
                Each tensor is in shape (N, num_classes * num_anchors, H, W)
            labels_list (list[Tensor]): The label that each anchor is assigned
                to. Shape (N * H * W * num_anchors, )
            pos_inds (list[Tensor]): List of bool tensors indicating whether
                the anchor is assigned to a positive label.
                Shape (N * H * W * num_anchors, )

        Returns:
            Tensor: A single float number indicating the positive recall.
        """
        with torch.no_grad():
            num_class = self.num_classes
            scores = [
                cls.permute(0, 2, 3, 1).reshape(-1, num_class)[pos]
                for cls, pos in zip(cls_scores, pos_inds)
            ]
            labels = [
                label.reshape(-1)[pos]
                for label, pos in zip(labels_list, pos_inds)
            ]
            scores = torch.cat(scores, dim=0)
            labels = torch.cat(labels, dim=0)
            if self.use_sigmoid_cls:
                scores = scores.sigmoid()
            else:
                scores = scores.softmax(dim=1)

            return accuracy(scores, labels, thresh=self.score_threshold)

    def collect_loss_level_single(self, cls_loss: Tensor, reg_loss: Tensor,
                                  assigned_gt_inds: Tensor,
                                  labels_seq: Tensor) -> Tensor:
        """Get the average loss in each FPN level w.r.t. each gt label.

        Args:
            cls_loss (Tensor): Classification loss of each feature map pixel,
              shape (num_anchor, num_class)
            reg_loss (Tensor): Regression loss of each feature map pixel,
              shape (num_anchor, 4)
            assigned_gt_inds (Tensor): It indicates which gt the prior is
              assigned to (0-based, -1: no assignment). shape (num_anchor),
            labels_seq: The rank of labels. shape (num_gt)

        Returns:
            Tensor: shape (num_gt), average loss of each gt in this level
        """
        if len(reg_loss.shape) == 2:  # iou loss has shape (num_prior, 4)
            reg_loss = reg_loss.sum(dim=-1)  # sum loss in tblr dims
        if len(cls_loss.shape) == 2:
            cls_loss = cls_loss.sum(dim=-1)  # sum loss in class dims
        loss = cls_loss + reg_loss
        assert loss.size(0) == assigned_gt_inds.size(0)
        # Default loss value is 1e6 for a layer where no anchor is positive
        #  to ensure it will not be chosen to back-propagate gradient
        losses_ = loss.new_full(labels_seq.shape, 1e6)
        for i, l in enumerate(labels_seq):
            match = assigned_gt_inds == l
            if match.any():
                losses_[i] = loss[match].mean()
        return losses_,

    def reweight_loss_single(self, cls_loss: Tensor, reg_loss: Tensor,
                             assigned_gt_inds: Tensor, labels: Tensor,
                             level: int, min_levels: Tensor) -> tuple:
        """Reweight loss values at each level.

        Reassign loss values at each level by masking those where the
        pre-calculated loss is too large. Then return the reduced losses.

        Args:
            cls_loss (Tensor): Element-wise classification loss.
              Shape: (num_anchors, num_classes)
            reg_loss (Tensor): Element-wise regression loss.
              Shape: (num_anchors, 4)
            assigned_gt_inds (Tensor): The gt indices that each anchor bbox
              is assigned to. -1 denotes a negative anchor, otherwise it is the
              gt index (0-based). Shape: (num_anchors, ),
            labels (Tensor): Label assigned to anchors. Shape: (num_anchors, ).
            level (int): The current level index in the pyramid
              (0-4 for RetinaNet)
            min_levels (Tensor): The best-matching level for each gt.
              Shape: (num_gts, ),

        Returns:
            tuple:

            - cls_loss: Reduced corrected classification loss. Scalar.
            - reg_loss: Reduced corrected regression loss. Scalar.
            - pos_flags (Tensor): Corrected bool tensor indicating the \
            final positive anchors. Shape: (num_anchors, ).
        """
        loc_weight = torch.ones_like(reg_loss)
        cls_weight = torch.ones_like(cls_loss)
        pos_flags = assigned_gt_inds >= 0  # positive pixel flag
        pos_indices = torch.nonzero(pos_flags, as_tuple=False).flatten()

        if pos_flags.any():  # pos pixels exist
            pos_assigned_gt_inds = assigned_gt_inds[pos_flags]
            zeroing_indices = (min_levels[pos_assigned_gt_inds] != level)
            neg_indices = pos_indices[zeroing_indices]

            if neg_indices.numel():
                pos_flags[neg_indices] = 0
                loc_weight[neg_indices] = 0
                # Only the weight corresponding to the label is
                #  zeroed out if not selected
                zeroing_labels = labels[neg_indices]
                assert (zeroing_labels >= 0).all()
                cls_weight[neg_indices, zeroing_labels] = 0

        # Weighted loss for both cls and reg loss
        cls_loss = weight_reduce_loss(cls_loss, cls_weight, reduction='sum')
        reg_loss = weight_reduce_loss(reg_loss, loc_weight, reduction='sum')

        return cls_loss, reg_loss, pos_flags
