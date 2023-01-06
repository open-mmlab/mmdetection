# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.model import ModuleList
from mmengine.structures import InstanceData
from torch import Tensor
from torch.autograd.function import Function

from mmdet.models.layers import multiclass_nms
from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.models.utils.misc import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor, scale_boxes
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptMultiConfig)


class _ScaleGradient(Function):

    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


@MODELS.register_module()
class CustomCascadeRoIHead(BaseRoIHead):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages: int,
                 stage_loss_weights: Union[List[float], Tuple[float]],
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.bbox_coder = TASK_UTILS.build(bbox_head[-1]['bbox_coder'])
        super().__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    def init_bbox_head(self, bbox_roi_extractor: MultiConfig,
                       bbox_head: MultiConfig) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (:obj:`ConfigDict`, dict or list):
                Config of box roi extractor.
            bbox_head (:obj:`ConfigDict`, dict or list): Config
                of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(MODELS.build(roi_extractor))
            self.bbox_head.append(MODELS.build(head))

    def init_mask_head(self, mask_roi_extractor: MultiConfig,
                       mask_head: MultiConfig) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_head (dict): Config of mask in mask head.
            mask_roi_extractor (:obj:`ConfigDict`, dict or list):
                Config of mask roi extractor.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(MODELS.build(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(MODELS.build(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    TASK_UTILS.build(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    TASK_UTILS.build(
                        rcnn_train_cfg.sampler,
                        default_args=dict(context=self)))

    def _bbox_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

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
        bbox_feats = _ScaleGradient.apply(bbox_feats, 1.0 / 3)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, stage: int, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Run forward function and calculate loss for box head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

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
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_results.update(rois=rois)

        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg[stage])
        bbox_results.update(bbox_loss_and_target)

        return bbox_results

    def _mask_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
        """
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_preds = mask_head(mask_feats)

        mask_results = dict(mask_preds=mask_preds)
        return mask_results

    def mask_loss(self, stage: int, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList) -> dict:
        """Run forward function and calculate loss for mask head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)

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
        # TODO: May add a new function in baseroihead
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        num_imgs = len(batch_data_samples)
        losses = dict()
        results_list = rpn_results_list
        for stage in range(self.num_stages):
            self.current_stage = stage

            stage_loss_weight = self.stage_loss_weights[stage]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
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
            bbox_results = self.bbox_loss(stage, x, sampling_results)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self.mask_loss(stage, x, sampling_results,
                                              batch_gt_instances)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)

            # refine bboxes
            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                with torch.no_grad():
                    results_list = bbox_head.refine_bboxes(
                        sampling_results, bbox_results, batch_img_metas)
                    # Empty proposal
                    if results_list is None:
                        break
        return losses

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False,
                     **kwargs) -> InstanceList:
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
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head[-1].predict_box_type,
                num_classes=self.bbox_head[-1].num_classes,
                score_per_cls=rcnn_test_cfg is None)

        rois, cls_scores, bbox_preds = self._refine_roi(
            x=x,
            rois=rois,
            proposals=proposals,
            batch_img_metas=batch_img_metas,
            num_proposals_per_img=num_proposals_per_img,
            **kwargs)

        results_list = self.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            rcnn_test_cfg=rcnn_test_cfg)
        return results_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: List[InstanceData],
                     rescale: bool = False) -> List[InstanceData]:
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

        num_mask_rois_per_img = [len(res) for res in results_list]
        aug_masks = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, mask_rois)
            mask_preds = mask_results['mask_preds']
            # split batch mask prediction back to each image
            mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
            aug_masks.append([m.sigmoid().detach() for m in mask_preds])

        merged_masks = []
        for i in range(len(batch_img_metas)):
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

    def _refine_roi(self, x: Tuple[Tensor], rois: Tensor,
                    proposals: List[Tensor], batch_img_metas: List[dict],
                    num_proposals_per_img: Sequence[int], **kwargs) -> tuple:
        """Multi-stage refinement of RoI.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): shape (n, 5), [batch_ind, x1, y1, x2, y2]
            batch_img_metas (list[dict]): List of image information.
            num_proposals_per_img (sequence[int]): number of proposals
                in each image.

        Returns:
            tuple:

               - rois (Tensor): Refined RoI.
               - cls_scores (list[Tensor]): Average predicted
                   cls score per image.
               - bbox_preds (list[Tensor]): Bbox branch predictions
                   for the last stage of per image.
        """
        # "ms" in variable names means multi-stage
        ms_scores = []
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(
                stage=stage, x=x, rois=rois, **kwargs)

            # split batch bbox prediction back to each image
            cls_scores = bbox_results['cls_score']
            bbox_preds = bbox_results['bbox_pred']

            rois = rois.split(num_proposals_per_img, 0)
            cls_scores = cls_scores.split(num_proposals_per_img, 0)
            ms_scores.append(cls_scores)

            # some detector with_reg is False, bbox_preds will be None
            if bbox_preds is not None:
                # TODO move this to a sabl_roi_head
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_preds, torch.Tensor):
                    bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
                else:
                    bbox_preds = self.bbox_head[stage].bbox_pred_split(
                        bbox_preds, num_proposals_per_img)
            else:
                bbox_preds = (None, ) * len(batch_img_metas)

            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                if bbox_head.custom_activation:
                    cls_scores = [
                        bbox_head.loss_cls.get_activation(s)
                        for s in cls_scores
                    ]
                refine_rois_list = []
                for i in range(len(batch_img_metas)):
                    if rois[i].shape[0] > 0:
                        bbox_label = cls_scores[i][:, :-1].argmax(dim=1)
                        # Refactor `bbox_head.regress_by_class` to only accept
                        # box tensor without img_idx concatenated.
                        refined_bboxes = bbox_head.regress_by_class(
                            rois[i][:, 1:], bbox_label, bbox_preds[i],
                            batch_img_metas[i])
                        refined_bboxes = get_box_tensor(refined_bboxes)
                        refined_rois = torch.cat(
                            [rois[i][:, [0]], refined_bboxes], dim=1)
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # softmax each stage
        ms_scores = [[
            F.softmax(score[i], dim=-1) for i in range(len(batch_img_metas))
        ] for score in ms_scores]

        # average scores of each image by stages
        cls_scores = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(len(batch_img_metas))
        ]

        # multiple proposal scores
        cls_scores = [(cls_scores[i].t() * proposals[i][:, -1]).t()**0.5
                      for i in range(len(batch_img_metas))]

        return rois, cls_scores, bbox_preds

    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            rois (tuple[Tensor]): Tuple of boxes to be transformed.
                Each has shape  (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
                (num_boxes, num_classes + 1).
            bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
                has shape (num_boxes, num_classes * 4).
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]
        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = cls_score
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                cls_score,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results
