# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn.functional as F
from mmengine import MessageHub
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import InstanceList
from ..utils.misc import unfold_wo_center
from .condinst_head import CondInstBboxHead, CondInstMaskHead


@MODELS.register_module()
class BoxInstBboxHead(CondInstBboxHead):
    """BoxInst box head used in https://arxiv.org/abs/2012.02310."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


@MODELS.register_module()
class BoxInstMaskHead(CondInstMaskHead):
    """BoxInst mask head used in https://arxiv.org/abs/2012.02310.

    This head outputs the mask for BoxInst.

    Args:
        pairwise_size (dict): The size of neighborhood for each pixel.
            Defaults to 3.
        pairwise_dilation (int): The dilation of neighborhood for each pixel.
            Defaults to 2.
        warmup_iters (int): Warmup iterations for pair-wise loss.
            Defaults to 10000.
    """

    def __init__(self,
                 *arg,
                 pairwise_size: int = 3,
                 pairwise_dilation: int = 2,
                 warmup_iters: int = 10000,
                 **kwargs) -> None:
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.warmup_iters = warmup_iters
        super().__init__(*arg, **kwargs)

    def get_pairwise_affinity(self, mask_logits: Tensor) -> Tensor:
        """Compute the pairwise affinity for each pixel."""
        log_fg_prob = F.logsigmoid(mask_logits).unsqueeze(1)
        log_bg_prob = F.logsigmoid(-mask_logits).unsqueeze(1)

        log_fg_prob_unfold = unfold_wo_center(
            log_fg_prob,
            kernel_size=self.pairwise_size,
            dilation=self.pairwise_dilation)
        log_bg_prob_unfold = unfold_wo_center(
            log_bg_prob,
            kernel_size=self.pairwise_size,
            dilation=self.pairwise_dilation)

        # the probability of making the same prediction:
        # p_i * p_j + (1 - p_i) * (1 - p_j)
        # we compute the the probability in log space
        # to avoid numerical instability
        log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
        log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

        # TODO: Figure out the difference between it and directly sum
        max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
        log_same_prob = torch.log(
            torch.exp(log_same_fg_prob - max_) +
            torch.exp(log_same_bg_prob - max_)) + max_

        return -log_same_prob[:, 0]

    def loss_by_feat(self, mask_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict], positive_infos: InstanceList,
                     **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (list[Tensor]): List of predicted masks, each has
                shape (num_classes, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``masks``,
                and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of multiple images.
            positive_infos (List[:obj:``InstanceData``]): Information of
                positive samples of each image that are assigned in detection
                head.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert positive_infos is not None, \
            'positive_infos should not be None in `BoxInstMaskHead`'
        losses = dict()

        loss_mask_project = 0.
        loss_mask_pairwise = 0.
        num_imgs = len(mask_preds)
        total_pos = 0.
        avg_fatcor = 0.

        for idx in range(num_imgs):
            (mask_pred, pos_mask_targets, pos_pairwise_masks, num_pos) = \
                self._get_targets_single(
                mask_preds[idx], batch_gt_instances[idx],
                positive_infos[idx])
            # mask loss
            total_pos += num_pos
            if num_pos == 0 or pos_mask_targets is None:
                loss_project = mask_pred.new_zeros(1).mean()
                loss_pairwise = mask_pred.new_zeros(1).mean()
                avg_fatcor += 0.
            else:
                # compute the project term
                loss_project_x = self.loss_mask(
                    mask_pred.max(dim=1, keepdim=True)[0],
                    pos_mask_targets.max(dim=1, keepdim=True)[0],
                    reduction_override='none').sum()
                loss_project_y = self.loss_mask(
                    mask_pred.max(dim=2, keepdim=True)[0],
                    pos_mask_targets.max(dim=2, keepdim=True)[0],
                    reduction_override='none').sum()
                loss_project = loss_project_x + loss_project_y
                # compute the pairwise term
                pairwise_affinity = self.get_pairwise_affinity(mask_pred)
                avg_fatcor += pos_pairwise_masks.sum().clamp(min=1.0)
                loss_pairwise = (pairwise_affinity * pos_pairwise_masks).sum()

            loss_mask_project += loss_project
            loss_mask_pairwise += loss_pairwise

        if total_pos == 0:
            total_pos += 1  # avoid nan
        if avg_fatcor == 0:
            avg_fatcor += 1  # avoid nan
        loss_mask_project = loss_mask_project / total_pos
        loss_mask_pairwise = loss_mask_pairwise / avg_fatcor
        message_hub = MessageHub.get_current_instance()
        iter = message_hub.get_info('iter')
        warmup_factor = min(iter / float(self.warmup_iters), 1.0)
        loss_mask_pairwise *= warmup_factor

        losses.update(
            loss_mask_project=loss_mask_project,
            loss_mask_pairwise=loss_mask_pairwise)
        return losses

    def _get_targets_single(self, mask_preds: Tensor,
                            gt_instances: InstanceData,
                            positive_info: InstanceData):
        """Compute targets for predictions of single image.

        Args:
            mask_preds (Tensor): Predicted prototypes with shape
                (num_classes, H, W).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes``, ``labels``,
                and ``masks`` attributes.
            positive_info (:obj:`InstanceData`): Information of positive
                samples that are assigned in detection head. It usually
                contains following keys.

                    - pos_assigned_gt_inds (Tensor): Assigner GT indexes of
                      positive proposals, has shape (num_pos, )
                    - pos_inds (Tensor): Positive index of image, has
                      shape (num_pos, ).
                    - param_pred (Tensor): Positive param preditions
                      with shape (num_pos, num_params).

        Returns:
            tuple: Usually returns a tuple containing learning targets.

            - mask_preds (Tensor): Positive predicted mask with shape
              (num_pos, mask_h, mask_w).
            - pos_mask_targets (Tensor): Positive mask targets with shape
              (num_pos, mask_h, mask_w).
            - pos_pairwise_masks (Tensor): Positive pairwise masks with
              shape: (num_pos, num_neighborhood, mask_h, mask_w).
            - num_pos (int): Positive numbers.
        """
        gt_bboxes = gt_instances.bboxes
        device = gt_bboxes.device
        # Note that gt_masks are generated by full box
        # from BoxInstDataPreprocessor
        gt_masks = gt_instances.masks.to_tensor(
            dtype=torch.bool, device=device).float()
        # Note that pairwise_masks are generated by image color similarity
        # from BoxInstDataPreprocessor
        pairwise_masks = gt_instances.pairwise_masks
        pairwise_masks = pairwise_masks.to(device=device)

        # process with mask targets
        pos_assigned_gt_inds = positive_info.get('pos_assigned_gt_inds')
        scores = positive_info.get('scores')
        centernesses = positive_info.get('centernesses')
        num_pos = pos_assigned_gt_inds.size(0)

        if gt_masks.size(0) == 0 or num_pos == 0:
            return mask_preds, None, None, 0
        # Since we're producing (near) full image masks,
        # it'd take too much vram to backprop on every single mask.
        # Thus we select only a subset.
        if (self.max_masks_to_train != -1) and \
           (num_pos > self.max_masks_to_train):
            perm = torch.randperm(num_pos)
            select = perm[:self.max_masks_to_train]
            mask_preds = mask_preds[select]
            pos_assigned_gt_inds = pos_assigned_gt_inds[select]
            num_pos = self.max_masks_to_train
        elif self.topk_masks_per_img != -1:
            unique_gt_inds = pos_assigned_gt_inds.unique()
            num_inst_per_gt = max(
                int(self.topk_masks_per_img / len(unique_gt_inds)), 1)

            keep_mask_preds = []
            keep_pos_assigned_gt_inds = []
            for gt_ind in unique_gt_inds:
                per_inst_pos_inds = (pos_assigned_gt_inds == gt_ind)
                mask_preds_per_inst = mask_preds[per_inst_pos_inds]
                gt_inds_per_inst = pos_assigned_gt_inds[per_inst_pos_inds]
                if sum(per_inst_pos_inds) > num_inst_per_gt:
                    per_inst_scores = scores[per_inst_pos_inds].sigmoid().max(
                        dim=1)[0]
                    per_inst_centerness = centernesses[
                        per_inst_pos_inds].sigmoid().reshape(-1, )
                    select = (per_inst_scores * per_inst_centerness).topk(
                        k=num_inst_per_gt, dim=0)[1]
                    mask_preds_per_inst = mask_preds_per_inst[select]
                    gt_inds_per_inst = gt_inds_per_inst[select]
                keep_mask_preds.append(mask_preds_per_inst)
                keep_pos_assigned_gt_inds.append(gt_inds_per_inst)
            mask_preds = torch.cat(keep_mask_preds)
            pos_assigned_gt_inds = torch.cat(keep_pos_assigned_gt_inds)
            num_pos = pos_assigned_gt_inds.size(0)

        # Follow the origin implement
        start = int(self.mask_out_stride // 2)
        gt_masks = gt_masks[:, start::self.mask_out_stride,
                            start::self.mask_out_stride]
        gt_masks = gt_masks.gt(0.5).float()
        pos_mask_targets = gt_masks[pos_assigned_gt_inds]
        pos_pairwise_masks = pairwise_masks[pos_assigned_gt_inds]
        pos_pairwise_masks = pos_pairwise_masks * pos_mask_targets.unsqueeze(1)

        return (mask_preds, pos_mask_targets, pos_pairwise_masks, num_pos)
