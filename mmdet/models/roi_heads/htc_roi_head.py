# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList, OptConfigType
from ..layers import adaptive_avg_pool2d
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .cascade_roi_head import CascadeRoIHead


@MODELS.register_module()
class HybridTaskCascadeRoIHead(CascadeRoIHead):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518

    Args:
        num_stages (int): Number of cascade stages.
        stage_loss_weights (list[float]): Loss weight for every stage.
        semantic_roi_extractor (:obj:`ConfigDict` or dict, optional):
            Config of semantic roi extractor. Defaults to None.
        Semantic_head (:obj:`ConfigDict` or dict, optional):
            Config of semantic head. Defaults to None.
        interleaved (bool): Whether to interleaves the box branch and mask
            branch. If True, the mask branch can take the refined bounding
            box predictions. Defaults to True.
        mask_info_flow (bool): Whether to turn on the mask information flow,
            which means that feeding the mask features of the preceding stage
            to the current stage. Defaults to True.
    """

    def __init__(self,
                 num_stages: int,
                 stage_loss_weights: List[float],
                 semantic_roi_extractor: OptConfigType = None,
                 semantic_head: OptConfigType = None,
                 semantic_fusion: Tuple[str] = ('bbox', 'mask'),
                 interleaved: bool = True,
                 mask_info_flow: bool = True,
                 **kwargs) -> None:
        super().__init__(
            num_stages=num_stages,
            stage_loss_weights=stage_loss_weights,
            **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            self.semantic_roi_extractor = MODELS.build(semantic_roi_extractor)
            self.semantic_head = MODELS.build(semantic_head)

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow

    # TODO move to base_roi_head later
    @property
    def with_semantic(self) -> bool:
        """bool: whether the head has semantic head"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    def _bbox_forward(
            self,
            stage: int,
            x: Tuple[Tensor],
            rois: Tensor,
            semantic_feat: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Box head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            semantic_feat (Tensor, optional): Semantic feature. Defaults to
                None.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def bbox_loss(self,
                  stage: int,
                  x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  semantic_feat: Optional[Tensor] = None) -> dict:
        """Run forward function and calculate loss for box head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            semantic_feat (Tensor, optional): Semantic feature. Defaults to
                None.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
                - `rois` (Tensor): RoIs with the shape (n, 5) where the first
                  column indicates batch id of each RoI.
                - `bbox_targets` (tuple):  Ground truth for proposals in a
                  single image. Containing the following list of Tensors:
                  (labels, label_weights, bbox_targets, bbox_weights)
        """
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(
            stage, x, rois, semantic_feat=semantic_feat)
        bbox_results.update(rois=rois)

        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg[stage])
        bbox_results.update(bbox_loss_and_target)
        return bbox_results

    def _mask_forward(self,
                      stage: int,
                      x: Tuple[Tensor],
                      rois: Tensor,
                      semantic_feat: Optional[Tensor] = None,
                      training: bool = True) -> Dict[str, Tensor]:
        """Mask head forward function used only in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            semantic_feat (Tensor, optional): Semantic feature. Defaults to
                None.
            training (bool): Mask Forward is different between training and
                testing. If True, use the mask forward in training.
                Defaults to True.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
        """
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats = mask_feats + mask_semantic_feat

        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if training:
            if self.mask_info_flow:
                last_feat = None
                for i in range(stage):
                    last_feat = self.mask_head[i](
                        mask_feats, last_feat, return_logits=False)
                mask_preds = mask_head(
                    mask_feats, last_feat, return_feat=False)
            else:
                mask_preds = mask_head(mask_feats, return_feat=False)

            mask_results = dict(mask_preds=mask_preds)
        else:
            aug_masks = []
            last_feat = None
            for i in range(self.num_stages):
                mask_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_preds, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_preds = mask_head(mask_feats)
            aug_masks.append(mask_preds)

            mask_results = dict(mask_preds=aug_masks)

        return mask_results

    def mask_loss(self,
                  stage: int,
                  x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList,
                  semantic_feat: Optional[Tensor] = None) -> dict:
        """Run forward function and calculate loss for mask head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            semantic_feat (Tensor, optional): Semantic feature. Defaults to
                None.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(
            stage=stage,
            x=x,
            rois=pos_rois,
            semantic_feat=semantic_feat,
            training=True)

        mask_head = self.mask_head[stage]
        mask_loss_and_target = mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[stage])
        mask_results.update(mask_loss_and_target)

        return mask_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
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
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        losses = dict()
        if self.with_semantic:
            gt_semantic_segs = [
                data_sample.gt_sem_seg.sem_seg
                for data_sample in batch_data_samples
            ]
            gt_semantic_segs = torch.stack(gt_semantic_segs)
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_segs)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None

        results_list = rpn_results_list
        num_imgs = len(batch_img_metas)
        for stage in range(self.num_stages):
            self.current_stage = stage

            stage_loss_weight = self.stage_loss_weights[stage]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[stage]
            bbox_sampler = self.bbox_sampler[stage]
            for i in range(num_imgs):
                results = results_list[i]
                # rename rpn_results.bboxes to rpn_results.priors
                if 'bboxes' in results:
                    results.priors = results.pop('bboxes')

                assign_result = bbox_assigner.assign(
                    results, batch_gt_instances[i],
                    batch_gt_instances_ignore[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    results,
                    batch_gt_instances[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self.bbox_loss(
                stage=stage,
                x=x,
                sampling_results=sampling_results,
                semantic_feat=semantic_feat)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    bbox_head = self.bbox_head[stage]
                    with torch.no_grad():
                        results_list = bbox_head.refine_bboxes(
                            sampling_results, bbox_results, batch_img_metas)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for i in range(num_imgs):
                            results = results_list[i]
                            # rename rpn_results.bboxes to rpn_results.priors
                            results.priors = results.pop('bboxes')
                            assign_result = bbox_assigner.assign(
                                results, batch_gt_instances[i],
                                batch_gt_instances_ignore[i])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                results,
                                batch_gt_instances[i],
                                feats=[lvl_feat[i][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                mask_results = self.mask_loss(
                    stage=stage,
                    x=x,
                    sampling_results=sampling_results,
                    batch_gt_instances=batch_gt_instances,
                    semantic_feat=semantic_feat)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if stage < self.num_stages - 1 and not self.interleaved:
                bbox_head = self.bbox_head[stage]
                with torch.no_grad():
                    results_list = bbox_head.refine_bboxes(
                        sampling_results=sampling_results,
                        bbox_results=bbox_results,
                        batch_img_metas=batch_img_metas)

        return losses

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results to
                the original image. Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        # TODO: nms_op in mmcv need be enhanced, the bbox result may get
        #  difference when not rescale in bbox_head

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x=x,
            semantic_feat=semantic_feat,
            batch_img_metas=batch_img_metas,
            rpn_results_list=rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        if self.with_mask:
            results_list = self.predict_mask(
                x=x,
                semantic_heat=semantic_feat,
                batch_img_metas=batch_img_metas,
                results_list=results_list,
                rescale=rescale)

        return results_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     semantic_heat: Tensor,
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            semantic_feat (Tensor): Semantic feature.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
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
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        num_imgs = len(batch_img_metas)
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas=batch_img_metas,
                device=mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_results = self._mask_forward(
            stage=-1,
            x=x,
            rois=mask_rois,
            semantic_feat=semantic_heat,
            training=False)
        # split batch mask prediction back to each image
        aug_masks = [[
            mask.sigmoid().detach()
            for mask in mask_preds.split(num_mask_rois_per_img, 0)
        ] for mask_preds in mask_results['mask_preds']]

        merged_masks = []
        for i in range(num_imgs):
            aug_mask = [mask[i] for mask in aug_masks]
            merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
            merged_masks.append(merged_mask)

        results_list = self.mask_head[-1].predict_by_feat(
            mask_preds=merged_masks,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
            activate_map=True)

        return results_list

    def forward(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
                batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        num_imgs = len(batch_img_metas)

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            rois, cls_scores, bbox_preds = self._refine_roi(
                x=x,
                rois=rois,
                semantic_feat=semantic_feat,
                batch_img_metas=batch_img_metas,
                num_proposals_per_img=num_proposals_per_img)
            results = results + (cls_scores, bbox_preds)
        # mask head
        if self.with_mask:
            rois = torch.cat(rois)
            mask_results = self._mask_forward(
                stage=-1,
                x=x,
                rois=rois,
                semantic_feat=semantic_feat,
                training=False)
            aug_masks = [[
                mask.sigmoid().detach()
                for mask in mask_preds.split(num_proposals_per_img, 0)
            ] for mask_preds in mask_results['mask_preds']]

            merged_masks = []
            for i in range(num_imgs):
                aug_mask = [mask[i] for mask in aug_masks]
                merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
                merged_masks.append(merged_mask)
            results = results + (merged_masks, )
        return results
