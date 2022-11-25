# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class MultiInstanceRoIHead(StandardRoIHead):
    """The roi head for Multi-instance prediction."""

    def __init__(self, num_instance: int = 2, *args, **kwargs) -> None:
        self.num_instance = num_instance
        super().__init__(*args, **kwargs)

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
                       bbox_head: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `cls_score_ref` (Tensor): The cls_score after refine model.
                - `bbox_pred_ref` (Tensor): The bbox_pred after refine model.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_results = self.bbox_head(bbox_feats)

        if self.bbox_head.with_refine:
            bbox_results = dict(
                cls_score=bbox_results[0],
                bbox_pred=bbox_results[1],
                cls_score_ref=bbox_results[2],
                bbox_pred_ref=bbox_results[3],
                bbox_feats=bbox_feats)
        else:
            bbox_results = dict(
                cls_score=bbox_results[0],
                bbox_pred=bbox_results[1],
                bbox_feats=bbox_feats)

        return bbox_results

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

        # If there is a refining process, add refine loss.
        if 'cls_score_ref' in bbox_results:
            bbox_loss_and_target = self.bbox_head.loss_and_target(
                cls_score=bbox_results['cls_score'],
                bbox_pred=bbox_results['bbox_pred'],
                rois=rois,
                sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg)
            bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
            bbox_loss_and_target_ref = self.bbox_head.loss_and_target(
                cls_score=bbox_results['cls_score_ref'],
                bbox_pred=bbox_results['bbox_pred_ref'],
                rois=rois,
                sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg)
            bbox_results['loss_bbox']['loss_rcnn_emd_ref'] = \
                bbox_loss_and_target_ref['loss_bbox']['loss_rcnn_emd']
        else:
            bbox_loss_and_target = self.bbox_head.loss_and_target(
                cls_score=bbox_results['cls_score'],
                bbox_pred=bbox_results['bbox_pred'],
                rois=rois,
                sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg)
            bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])

        return bbox_results

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

        sampling_results = []
        for i in range(len(batch_data_samples)):
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
                batch_gt_instances_ignore=batch_gt_instances_ignore[i])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas, rois.device, task_type='bbox')

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        if 'cls_score_ref' in bbox_results:
            cls_scores = bbox_results['cls_score_ref']
            bbox_preds = bbox_results['bbox_pred_ref']
        else:
            cls_scores = bbox_results['cls_score']
            bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        if bbox_preds is not None:
            bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list
