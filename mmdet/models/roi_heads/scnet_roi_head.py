# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList, OptConfigType
from ..layers import adaptive_avg_pool2d
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .cascade_roi_head import CascadeRoIHead


@MODELS.register_module()
class SCNetRoIHead(CascadeRoIHead):
    """RoIHead for `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        num_stages (int): number of cascade stages.
        stage_loss_weights (list): loss weight of cascade stages.
        semantic_roi_extractor (dict): config to init semantic roi extractor.
        semantic_head (dict): config to init semantic head.
        feat_relay_head (dict): config to init feature_relay_head.
        glbctx_head (dict): config to init global context head.
    """

    def __init__(self,
                 num_stages: int,
                 stage_loss_weights: List[float],
                 semantic_roi_extractor: OptConfigType = None,
                 semantic_head: OptConfigType = None,
                 feat_relay_head: OptConfigType = None,
                 glbctx_head: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(
            num_stages=num_stages,
            stage_loss_weights=stage_loss_weights,
            **kwargs)
        assert self.with_bbox and self.with_mask
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            self.semantic_roi_extractor = MODELS.build(semantic_roi_extractor)
            self.semantic_head = MODELS.build(semantic_head)

        if feat_relay_head is not None:
            self.feat_relay_head = MODELS.build(feat_relay_head)

        if glbctx_head is not None:
            self.glbctx_head = MODELS.build(glbctx_head)

    def init_mask_head(self, mask_roi_extractor: ConfigType,
                       mask_head: ConfigType) -> None:
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)
            self.mask_head = MODELS.build(mask_head)

    # TODO move to base_roi_head later
    @property
    def with_semantic(self) -> bool:
        """bool: whether the head has semantic head"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    @property
    def with_feat_relay(self) -> bool:
        """bool: whether the head has feature relay head"""
        return (hasattr(self, 'feat_relay_head')
                and self.feat_relay_head is not None)

    @property
    def with_glbctx(self) -> bool:
        """bool: whether the head has global context head"""
        return hasattr(self, 'glbctx_head') and self.glbctx_head is not None

    def _fuse_glbctx(self, roi_feats: Tensor, glbctx_feat: Tensor,
                     rois: Tensor) -> Tensor:
        """Fuse global context feats with roi feats.

        Args:
            roi_feats (Tensor): RoI features.
            glbctx_feat (Tensor): Global context feature..
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
            Tensor: Fused feature.
        """
        assert roi_feats.size(0) == rois.size(0)
        # RuntimeError: isDifferentiableType(variable.scalar_type())
        # INTERNAL ASSERT FAILED if detach() is not used when calling
        # roi_head.predict().
        img_inds = torch.unique(rois[:, 0].detach().cpu(), sorted=True).long()
        fused_feats = torch.zeros_like(roi_feats)
        for img_id in img_inds:
            inds = (rois[:, 0] == img_id.item())
            fused_feats[inds] = roi_feats[inds] + glbctx_feat[img_id]
        return fused_feats

    def _slice_pos_feats(self, feats: Tensor,
                         sampling_results: List[SamplingResult]) -> Tensor:
        """Get features from pos rois.

        Args:
            feats (Tensor): Input features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            Tensor: Sliced features.
        """
        num_rois = [res.priors.size(0) for res in sampling_results]
        num_pos_rois = [res.pos_priors.size(0) for res in sampling_results]
        inds = torch.zeros(sum(num_rois), dtype=torch.bool)
        start = 0
        for i in range(len(num_rois)):
            start = 0 if i == 0 else start + num_rois[i - 1]
            stop = start + num_pos_rois[i]
            inds[start:stop] = 1
        sliced_feats = feats[inds]
        return sliced_feats

    def _bbox_forward(self,
                      stage: int,
                      x: Tuple[Tensor],
                      rois: Tensor,
                      semantic_feat: Optional[Tensor] = None,
                      glbctx_feat: Optional[Tensor] = None) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            semantic_feat (Tensor): Semantic feature. Defaults to None.
            glbctx_feat (Tensor): Global context feature. Defaults to None.

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
        if self.with_semantic and semantic_feat is not None:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        if self.with_glbctx and glbctx_feat is not None:
            bbox_feats = self._fuse_glbctx(bbox_feats, glbctx_feat, rois)
        cls_score, bbox_pred, relayed_feat = bbox_head(
            bbox_feats, return_shared_feat=True)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            relayed_feat=relayed_feat)
        return bbox_results

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor,
                      semantic_feat: Optional[Tensor] = None,
                      glbctx_feat: Optional[Tensor] = None,
                      relayed_feat: Optional[Tensor] = None) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            semantic_feat (Tensor): Semantic feature. Defaults to None.
            glbctx_feat (Tensor): Global context feature. Defaults to None.
            relayed_feat (Tensor): Relayed feature. Defaults to None.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
        """
        mask_feats = self.mask_roi_extractor(
            x[:self.mask_roi_extractor.num_inputs], rois)
        if self.with_semantic and semantic_feat is not None:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.with_glbctx and glbctx_feat is not None:
            mask_feats = self._fuse_glbctx(mask_feats, glbctx_feat, rois)
        if self.with_feat_relay and relayed_feat is not None:
            mask_feats = mask_feats + relayed_feat
        mask_preds = self.mask_head(mask_feats)
        mask_results = dict(mask_preds=mask_preds)

        return mask_results

    def bbox_loss(self,
                  stage: int,
                  x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  semantic_feat: Optional[Tensor] = None,
                  glbctx_feat: Optional[Tensor] = None) -> dict:
        """Run forward function and calculate loss for box head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            semantic_feat (Tensor): Semantic feature. Defaults to None.
            glbctx_feat (Tensor): Global context feature. Defaults to None.

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
            stage,
            x,
            rois,
            semantic_feat=semantic_feat,
            glbctx_feat=glbctx_feat)
        bbox_results.update(rois=rois)

        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg[stage])

        bbox_results.update(bbox_loss_and_target)
        return bbox_results

    def mask_loss(self,
                  x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList,
                  semantic_feat: Optional[Tensor] = None,
                  glbctx_feat: Optional[Tensor] = None,
                  relayed_feat: Optional[Tensor] = None) -> dict:
        """Run forward function and calculate loss for mask head in training.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            semantic_feat (Tensor): Semantic feature. Defaults to None.
            glbctx_feat (Tensor): Global context feature. Defaults to None.
            relayed_feat (Tensor): Relayed feature. Defaults to None.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(
            x,
            pos_rois,
            semantic_feat=semantic_feat,
            glbctx_feat=glbctx_feat,
            relayed_feat=relayed_feat)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[-1])
        mask_results.update(mask_loss_and_target)

        return mask_results

    def semantic_loss(self, x: Tuple[Tensor],
                      batch_data_samples: SampleList) -> dict:
        """Semantic segmentation loss.

        Args:
            x (Tuple[Tensor]): Tuple of multi-level img features.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `semantic_feat` (Tensor): Semantic feature.
                - `loss_seg` (dict): Semantic segmentation loss.
        """
        gt_semantic_segs = [
            data_sample.gt_sem_seg.sem_seg
            for data_sample in batch_data_samples
        ]
        gt_semantic_segs = torch.stack(gt_semantic_segs)
        semantic_pred, semantic_feat = self.semantic_head(x)
        loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_segs)

        semantic_results = dict(loss_seg=loss_seg, semantic_feat=semantic_feat)

        return semantic_results

    def global_context_loss(self, x: Tuple[Tensor],
                            batch_gt_instances: InstanceList) -> dict:
        """Global context loss.

        Args:
            x (Tuple[Tensor]): Tuple of multi-level img features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `glbctx_feat` (Tensor): Global context feature.
                - `loss_glbctx` (dict): Global context loss.
        """
        gt_labels = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        mc_pred, glbctx_feat = self.glbctx_head(x)
        loss_glbctx = self.glbctx_head.loss(mc_pred, gt_labels)
        global_context_results = dict(
            loss_glbctx=loss_glbctx, glbctx_feat=glbctx_feat)

        return global_context_results

    def loss(self, x: Tensor, rpn_results_list: InstanceList,
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

        losses = dict()

        # semantic segmentation branch
        if self.with_semantic:
            semantic_results = self.semantic_loss(
                x=x, batch_data_samples=batch_data_samples)
            losses['loss_semantic_seg'] = semantic_results['loss_seg']
            semantic_feat = semantic_results['semantic_feat']
        else:
            semantic_feat = None

        # global context branch
        if self.with_glbctx:
            global_context_results = self.global_context_loss(
                x=x, batch_gt_instances=batch_gt_instances)
            losses['loss_glbctx'] = global_context_results['loss_glbctx']
            glbctx_feat = global_context_results['glbctx_feat']
        else:
            glbctx_feat = None

        results_list = rpn_results_list
        num_imgs = len(batch_img_metas)
        for stage in range(self.num_stages):
            stage_loss_weight = self.stage_loss_weights[stage]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[stage]
            bbox_sampler = self.bbox_sampler[stage]
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

            # bbox head forward and loss
            bbox_results = self.bbox_loss(
                stage=stage,
                x=x,
                sampling_results=sampling_results,
                semantic_feat=semantic_feat,
                glbctx_feat=glbctx_feat)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            # refine bboxes
            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                with torch.no_grad():
                    results_list = bbox_head.refine_bboxes(
                        sampling_results=sampling_results,
                        bbox_results=bbox_results,
                        batch_img_metas=batch_img_metas)

        if self.with_feat_relay:
            relayed_feat = self._slice_pos_feats(bbox_results['relayed_feat'],
                                                 sampling_results)
            relayed_feat = self.feat_relay_head(relayed_feat)
        else:
            relayed_feat = None

        # mask head forward and loss
        mask_results = self.mask_loss(
            x=x,
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            semantic_feat=semantic_feat,
            glbctx_feat=glbctx_feat,
            relayed_feat=relayed_feat)
        mask_stage_loss_weight = sum(self.stage_loss_weights)
        losses['loss_mask'] = mask_stage_loss_weight * mask_results[
            'loss_mask']['loss_mask']

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

        if self.with_glbctx:
            _, glbctx_feat = self.glbctx_head(x)
        else:
            glbctx_feat = None

        # TODO: nms_op in mmcv need be enhanced, the bbox result may get
        #  difference when not rescale in bbox_head

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x=x,
            semantic_feat=semantic_feat,
            glbctx_feat=glbctx_feat,
            batch_img_metas=batch_img_metas,
            rpn_results_list=rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        if self.with_mask:
            results_list = self.predict_mask(
                x=x,
                semantic_heat=semantic_feat,
                glbctx_feat=glbctx_feat,
                batch_img_metas=batch_img_metas,
                results_list=results_list,
                rescale=rescale)

        return results_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     semantic_heat: Tensor,
                     glbctx_feat: Tensor,
                     batch_img_metas: List[dict],
                     results_list: List[InstanceData],
                     rescale: bool = False) -> List[InstanceData]:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            semantic_feat (Tensor): Semantic feature.
            glbctx_feat (Tensor): Global context feature.
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

        bboxes_results = self._bbox_forward(
            stage=-1,
            x=x,
            rois=mask_rois,
            semantic_feat=semantic_heat,
            glbctx_feat=glbctx_feat)
        relayed_feat = bboxes_results['relayed_feat']
        relayed_feat = self.feat_relay_head(relayed_feat)

        mask_results = self._mask_forward(
            x=x,
            rois=mask_rois,
            semantic_feat=semantic_heat,
            glbctx_feat=glbctx_feat,
            relayed_feat=relayed_feat)
        mask_preds = mask_results['mask_preds']

        # split batch mask prediction back to each image
        num_bbox_per_img = tuple(len(_bbox) for _bbox in bboxes)
        mask_preds = mask_preds.split(num_bbox_per_img, 0)

        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)

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

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        if self.with_glbctx:
            _, glbctx_feat = self.glbctx_head(x)
        else:
            glbctx_feat = None

        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            rois, cls_scores, bbox_preds = self._refine_roi(
                x=x,
                rois=rois,
                semantic_feat=semantic_feat,
                glbctx_feat=glbctx_feat,
                batch_img_metas=batch_img_metas,
                num_proposals_per_img=num_proposals_per_img)
            results = results + (cls_scores, bbox_preds)
        # mask head
        if self.with_mask:
            rois = torch.cat(rois)
            bboxes_results = self._bbox_forward(
                stage=-1,
                x=x,
                rois=rois,
                semantic_feat=semantic_feat,
                glbctx_feat=glbctx_feat)
            relayed_feat = bboxes_results['relayed_feat']
            relayed_feat = self.feat_relay_head(relayed_feat)
            mask_results = self._mask_forward(
                x=x,
                rois=rois,
                semantic_feat=semantic_feat,
                glbctx_feat=glbctx_feat,
                relayed_feat=relayed_feat)
            mask_preds = mask_results['mask_preds']
            mask_preds = mask_preds.split(num_proposals_per_img, 0)
            results = results + (mask_preds, )
        return results
