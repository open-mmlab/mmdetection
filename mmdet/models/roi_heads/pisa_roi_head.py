# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from torch import Tensor

from mmdet.models.task_modules import SamplingResult
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList
from ..losses.pisa_loss import carl_loss, isr_p
from ..utils import unpack_gt_instances
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class PISARoIHead(StandardRoIHead):
    r"""The RoI head for `Prime Sample Attention in Object Detection
    <https://arxiv.org/abs/1904.04821>`_."""

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        neg_label_weights = []
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
            if isinstance(sampling_result, tuple):
                sampling_result, neg_label_weight = sampling_result
            sampling_results.append(sampling_result)
            neg_label_weights.append(neg_label_weight)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(
                x, sampling_results, neg_label_weights=neg_label_weights)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        return losses

    def bbox_loss(self,
                  x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  neg_label_weights: List[Tensor] = None) -> dict:
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
        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  self.train_cfg)

        # neg_label_weights obtained by sampler is image-wise, mapping back to
        # the corresponding location in label weights
        if neg_label_weights[0] is not None:
            label_weights = bbox_targets[1]
            cur_num_rois = 0
            for i in range(len(sampling_results)):
                num_pos = sampling_results[i].pos_inds.size(0)
                num_neg = sampling_results[i].neg_inds.size(0)
                label_weights[cur_num_rois + num_pos:cur_num_rois + num_pos +
                              num_neg] = neg_label_weights[i]
                cur_num_rois += num_pos + num_neg

        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Apply ISR-P
        isr_cfg = self.train_cfg.get('isr', None)
        if isr_cfg is not None:
            bbox_targets = isr_p(
                cls_score,
                bbox_pred,
                bbox_targets,
                rois,
                sampling_results,
                self.bbox_head.loss_cls,
                self.bbox_head.bbox_coder,
                **isr_cfg,
                num_class=self.bbox_head.num_classes)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, rois,
                                        *bbox_targets)

        # Add CARL Loss
        carl_cfg = self.train_cfg.get('carl', None)
        if carl_cfg is not None:
            loss_carl = carl_loss(
                cls_score,
                bbox_targets[0],
                bbox_pred,
                bbox_targets[2],
                self.bbox_head.loss_bbox,
                **carl_cfg,
                num_class=self.bbox_head.num_classes)
            loss_bbox.update(loss_carl)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
