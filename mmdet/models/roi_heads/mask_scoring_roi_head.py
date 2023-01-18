# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils.misc import empty_instances
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class MaskScoringRoIHead(StandardRoIHead):
    """Mask Scoring RoIHead for `Mask Scoring RCNN.

    <https://arxiv.org/abs/1903.00241>`_.

    Args:
        mask_iou_head (:obj`ConfigDict`, dict): The config of mask_iou_head.
    """

    def __init__(self, mask_iou_head: ConfigType, **kwargs):
        assert mask_iou_head is not None
        super().__init__(**kwargs)
        self.mask_iou_head = MODELS.build(mask_iou_head)

    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
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
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            results = results + (bbox_results['cls_score'],
                                 bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            results = results + (mask_results['mask_preds'], )

            # mask iou head
            cls_score = bbox_results['cls_score'][:100]
            mask_preds = mask_results['mask_preds']
            mask_feats = mask_results['mask_feats']
            _, labels = cls_score[:, :self.bbox_head.num_classes].max(dim=1)
            mask_iou_preds = self.mask_iou_head(
                mask_feats, mask_preds[range(labels.size(0)), labels])
            results = results + (mask_iou_preds, )

        return results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats,
                  batch_gt_instances: InstanceList) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
                - `loss_mask_iou` (Tensor): mask iou loss.
        """
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)
        mask_targets = mask_loss_and_target['mask_targets']
        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        if mask_results['loss_mask'] is None:
            return mask_results

        # mask iou head forward and loss
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        pos_mask_pred = mask_results['mask_preds'][
            range(mask_results['mask_preds'].size(0)), pos_labels]
        mask_iou_pred = self.mask_iou_head(mask_results['mask_feats'],
                                           pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)),
                                          pos_labels]

        loss_mask_iou = self.mask_iou_head.loss_and_target(
            pos_mask_iou_pred, pos_mask_pred, mask_targets, sampling_results,
            batch_gt_instances, self.train_cfg)
        mask_results['loss_mask'].update(loss_mask_iou)
        return mask_results

    def predict_mask(self,
                     x: Tensor,
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
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
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results['mask_preds']
        mask_feats = mask_results['mask_feats']
        # get mask scores with mask iou head
        labels = torch.cat([res.labels for res in results_list])
        mask_iou_preds = self.mask_iou_head(
            mask_feats, mask_preds[range(labels.size(0)), labels])
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
        mask_iou_preds = mask_iou_preds.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        results_list = self.mask_iou_head.predict_by_feat(
            mask_iou_preds=mask_iou_preds, results_list=results_list)
        return results_list
