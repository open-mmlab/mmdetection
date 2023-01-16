# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils.misc import unpack_gt_instances
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class GridRoIHead(StandardRoIHead):
    """Implementation of `Grid RoI Head <https://arxiv.org/abs/1811.12030>`_

    Args:
        grid_roi_extractor (:obj:`ConfigDict` or dict): Config of
            roi extractor.
        grid_head (:obj:`ConfigDict` or dict): Config of grid head
    """

    def __init__(self, grid_roi_extractor: ConfigType, grid_head: ConfigType,
                 **kwargs) -> None:
        assert grid_head is not None
        super().__init__(**kwargs)
        if grid_roi_extractor is not None:
            self.grid_roi_extractor = MODELS.build(grid_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.grid_roi_extractor = self.bbox_roi_extractor
        self.grid_head = MODELS.build(grid_head)

    def _random_jitter(self,
                       sampling_results: List[SamplingResult],
                       batch_img_metas: List[dict],
                       amplitude: float = 0.15) -> List[SamplingResult]:
        """Ramdom jitter positive proposals for training.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_img_metas (list[dict]): List of image information.
            amplitude (float): Amplitude of random offset. Defaults to 0.15.

        Returns:
            list[obj:SamplingResult]: SamplingResults after random jittering.
        """
        for sampling_result, img_meta in zip(sampling_results,
                                             batch_img_metas):
            bboxes = sampling_result.pos_priors
            random_offsets = bboxes.new_empty(bboxes.shape[0], 4).uniform_(
                -amplitude, amplitude)
            # before jittering
            cxcy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
            wh = (bboxes[:, 2:4] - bboxes[:, :2]).abs()
            # after jittering
            new_cxcy = cxcy + wh * random_offsets[:, :2]
            new_wh = wh * (1 + random_offsets[:, 2:])
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            max_shape = img_meta['img_shape']
            if max_shape is not None:
                new_bboxes[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bboxes[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)

            sampling_result.pos_priors = new_bboxes
        return sampling_results

    # TODO: Forward is incorrect and need to refactor.
    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (Tuple[Tensor]): Multi-level features that may have different
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
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            results = results + (bbox_results['cls_score'], )
            if self.bbox_head.with_reg:
                results = results + (bbox_results['bbox_pred'], )

            # grid head
            grid_rois = rois[:100]
            grid_feats = self.grid_roi_extractor(
                x[:len(self.grid_roi_extractor.featmap_strides)], grid_rois)
            if self.with_shared_head:
                grid_feats = self.shared_head(grid_feats)
            self.grid_head.test_mode = True
            grid_preds = self.grid_head(grid_feats)
            results = results + (grid_preds, )

        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            results = results + (mask_results['mask_preds'], )
        return results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList, **kwargs) -> dict:
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
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

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
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results, batch_img_metas)
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
                  batch_img_metas: Optional[List[dict]] = None) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list[:obj:`SamplingResult`]): Sampling results.
            batch_img_metas (list[dict], optional): Meta information of each
                image, e.g., image size, scaling factor, etc.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

            - `cls_score` (Tensor): Classification scores.
            - `bbox_pred` (Tensor): Box energies / deltas.
            - `bbox_feats` (Tensor): Extract bbox RoI features.
            - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        assert batch_img_metas is not None
        bbox_results = super().bbox_loss(x, sampling_results)

        # Grid head forward and loss
        sampling_results = self._random_jitter(sampling_results,
                                               batch_img_metas)
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])

        # GN in head does not support zero shape input
        if pos_rois.shape[0] == 0:
            return bbox_results

        grid_feats = self.grid_roi_extractor(
            x[:self.grid_roi_extractor.num_inputs], pos_rois)
        if self.with_shared_head:
            grid_feats = self.shared_head(grid_feats)
        # Accelerate training
        max_sample_num_grid = self.train_cfg.get('max_num_grid', 192)
        sample_idx = torch.randperm(
            grid_feats.shape[0])[:min(grid_feats.shape[0], max_sample_num_grid
                                      )]
        grid_feats = grid_feats[sample_idx]
        grid_pred = self.grid_head(grid_feats)

        loss_grid = self.grid_head.loss(grid_pred, sample_idx,
                                        sampling_results, self.train_cfg)

        bbox_results['loss_bbox'].update(loss_grid)
        return bbox_results

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
            rcnn_test_cfg (:obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape \
            (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4), the last \
            dimension 4 arrange as (x1, y1, x2, y2).
        """
        results_list = super().predict_bbox(
            x,
            batch_img_metas=batch_img_metas,
            rpn_results_list=rpn_results_list,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=False)

        grid_rois = bbox2roi([res.bboxes for res in results_list])
        if grid_rois.shape[0] != 0:
            grid_feats = self.grid_roi_extractor(
                x[:len(self.grid_roi_extractor.featmap_strides)], grid_rois)
            if self.with_shared_head:
                grid_feats = self.shared_head(grid_feats)
            self.grid_head.test_mode = True
            grid_preds = self.grid_head(grid_feats)
            results_list = self.grid_head.predict_by_feat(
                grid_preds=grid_preds,
                results_list=results_list,
                batch_img_metas=batch_img_metas,
                rescale=rescale)

        return results_list
