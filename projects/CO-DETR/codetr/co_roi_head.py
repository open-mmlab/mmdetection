from typing import List, Tuple

import torch
from torch import Tensor

from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList


@MODELS.register_module()
class CoStandardRoIHead(StandardRoIHead):

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        max_proposal = 2000

        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
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
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

            bbox_targets = bbox_results['bbox_targets']
            for res in sampling_results:
                max_proposal = min(max_proposal, res.bboxes.shape[0])
            ori_coords = bbox2roi([res.bboxes for res in sampling_results])
            ori_proposals, ori_labels, \
                ori_bbox_targets, ori_bbox_feats = [], [], [], []
            for i in range(num_imgs):
                idx = (ori_coords[:, 0] == i).nonzero().squeeze(1)
                idx = idx[:max_proposal]
                ori_proposal = ori_coords[idx][:, 1:].unsqueeze(0)
                ori_label = bbox_targets[0][idx].unsqueeze(0)
                ori_bbox_target = bbox_targets[2][idx].unsqueeze(0)
                ori_bbox_feat = bbox_results['bbox_feats'].mean(-1).mean(-1)
                ori_bbox_feat = ori_bbox_feat[idx].unsqueeze(0)
                ori_proposals.append(ori_proposal)
                ori_labels.append(ori_label)
                ori_bbox_targets.append(ori_bbox_target)
                ori_bbox_feats.append(ori_bbox_feat)
            ori_coords = torch.cat(ori_proposals, dim=0)
            ori_labels = torch.cat(ori_labels, dim=0)
            ori_bbox_targets = torch.cat(ori_bbox_targets, dim=0)
            ori_bbox_feats = torch.cat(ori_bbox_feats, dim=0)
            pos_coords = (ori_coords, ori_labels, ori_bbox_targets,
                          ori_bbox_feats, 'rcnn')
            losses.update(pos_coords=pos_coords)

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
        # diff
        bbox_results.update(bbox_targets=bbox_loss_and_target['bbox_targets'])
        return bbox_results
