# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from mmdet.models.losses import SmoothL1Loss
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList
from ..utils.misc import unpack_gt_instances
from .standard_roi_head import StandardRoIHead

EPS = 1e-15


@MODELS.register_module()
class DynamicRoIHead(StandardRoIHead):
    """RoI head for `Dynamic R-CNN <https://arxiv.org/abs/2004.06002>`_."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(self.bbox_head.loss_bbox, SmoothL1Loss)
        # the IoU history of the past `update_iter_interval` iterations
        self.iou_history = []
        # the beta history of the past `update_iter_interval` iterations
        self.beta_history = []

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        """Forward function for training.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        cur_iou = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            # record the `iou_topk`-th largest IoU in an image
            iou_topk = min(self.train_cfg.dynamic_rcnn.iou_topk,
                           len(assign_result.max_overlaps))
            ious, _ = torch.topk(assign_result.max_overlaps, iou_topk)
            cur_iou.append(ious[-1].item())
            sampling_results.append(sampling_result)
        # average the current IoUs over images
        cur_iou = np.mean(cur_iou)
        self.iou_history.append(cur_iou)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        # update IoU threshold and SmoothL1 beta
        update_iter_interval = self.train_cfg.dynamic_rcnn.update_iter_interval
        if len(self.iou_history) % update_iter_interval == 0:
            new_iou_thr, new_beta = self.update_hyperparameters()

        return losses

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)
        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])

        # record the `beta_topk`-th smallest target
        # `bbox_targets[2]` and `bbox_targets[3]` stand for bbox_targets
        # and bbox_weights, respectively
        bbox_targets = bbox_loss_and_target['bbox_targets']
        pos_inds = bbox_targets[3][:, 0].nonzero().squeeze(1)
        num_pos = len(pos_inds)
        num_imgs = len(sampling_results)
        if num_pos > 0:
            cur_target = bbox_targets[2][pos_inds, :2].abs().mean(dim=1)
            beta_topk = min(self.train_cfg.dynamic_rcnn.beta_topk * num_imgs,
                            num_pos)
            cur_target = torch.kthvalue(cur_target, beta_topk)[0].item()
            self.beta_history.append(cur_target)

        return bbox_results

    def update_hyperparameters(self):
        """Update hyperparameters like IoU thresholds for assigner and beta for
        SmoothL1 loss based on the training statistics.

        Returns:
            tuple[float]: the updated ``iou_thr`` and ``beta``.
        """
        new_iou_thr = max(self.train_cfg.dynamic_rcnn.initial_iou,
                          np.mean(self.iou_history))
        self.iou_history = []
        self.bbox_assigner.pos_iou_thr = new_iou_thr
        self.bbox_assigner.neg_iou_thr = new_iou_thr
        self.bbox_assigner.min_pos_iou = new_iou_thr
        if (not self.beta_history) or (np.median(self.beta_history) < EPS):
            # avoid 0 or too small value for new_beta
            new_beta = self.bbox_head.loss_bbox.beta
        else:
            new_beta = min(self.train_cfg.dynamic_rcnn.initial_beta,
                           np.median(self.beta_history))
        self.beta_history = []
        self.bbox_head.loss_bbox.beta = new_beta
        return new_iou_thr, new_beta
