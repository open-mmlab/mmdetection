# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads import DINOHead
from mmdet.registry import MODELS
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import InstanceList
from .utils import KeysRecorder


@MODELS.register_module()
class AlignDETRHead(DINOHead):
    r"""Head of the Align-DETR: Improving DETR with Simple IoU-aware BCE loss

    Code is modified from the `official github repo
    <https://github.com/FelixCaae/AlignDETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2304.07527>`_ .

    Args:
        all_layers_num_gt_repeat List[int]: Number to repeat gt for 1-to-k
            matching between ground truth and predictions of each decoder
            layer. Only used for matching queries, not for denoising queries.
            Element count is `num_pred_layer`. If `as_two_stage` is True, then
            the last element is for encoder output, and the others for
            decoder layers. Otherwise, all elements are for decoder layers.
            Defaults to a list of `1` for the last decoder layer and `2` for
            the others.
        alpha (float): Hyper-parameter of classification loss that controls
            the proportion of each item to calculate `t`, the weighted
            geometric average of the confident score and the IoU score, to
            align classification and regression scores. Defaults to `0.25`.
        gamma (float): Hyper-parameter of classification loss to do the hard
            negative mining. Defaults to `2.0`.
        tau (float): Hyper-parameter of classification and regression losses,
            it is the temperature controlling the sharpness of the function
            to calculate positive sample weight. Defaults to `1.5`.
    """

    def __init__(self,
                 *args,
                 all_layers_num_gt_repeat: List[int] = None,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 tau: float = 1.5,
                 **kwargs) -> None:
        self.all_layers_num_gt_repeat = all_layers_num_gt_repeat
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.weight_table = torch.zeros(
            len(all_layers_num_gt_repeat), max(all_layers_num_gt_repeat))
        for layer_index, num_gt_repeat in enumerate(all_layers_num_gt_repeat):
            self.weight_table[layer_index][:num_gt_repeat] = torch.exp(
                -torch.arange(num_gt_repeat) / tau)

        super().__init__(*args, **kwargs)
        assert len(self.all_layers_num_gt_repeat) == self.num_pred_layer

    def loss_by_feat(self, all_layers_cls_scores: Tensor, *args,
                     **kwargs) -> Any:
        """Loss function.
            AlignDETR: This method is based on `DINOHead.loss_by_feat`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Wrap `all_layers_cls_scores` with KeysRecorder to record its
        #   `__getitem__` keys and get decoder layer index.
        all_layers_cls_scores = KeysRecorder(all_layers_cls_scores)
        result = super(AlignDETRHead,
                       self).loss_by_feat(all_layers_cls_scores, *args,
                                          **kwargs)
        return result

    def loss_by_feat_single(self, cls_scores: Union[KeysRecorder, Tensor],
                            bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.
            AlignDETR: This method is based on `DINOHead.loss_by_feat_single`.

        Args:
            cls_scores (Union[KeysRecorder, Tensor]): Box score logits from a
                single decoder layer for all images, has shape (bs,
                num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        # AlignDETR: Get layer_index.
        if isinstance(cls_scores, KeysRecorder):
            # Outputs are from decoder layer. Get layer_index from
            #   `__getitem__` keys history.
            keys = [key for key in cls_scores.keys if isinstance(key, int)]
            assert len(keys) == 1, \
                'Failed to extract key from cls_scores.keys: {}'.format(keys)
            layer_index = keys[0]
            # Get dn_cls_scores tensor.
            cls_scores = cls_scores.obj
        else:
            # Outputs are from encoder layer.
            layer_index = self.num_pred_layer - 1

        for img_meta in batch_img_metas:
            img_meta['layer_index'] = layer_index

        results = super(AlignDETRHead, self).loss_by_feat_single(
            cls_scores,
            bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)
        return results

    def get_targets(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.
        AlignDETR: This method is based on `DETRHead.get_targets`.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        results = super(AlignDETRHead,
                        self).get_targets(cls_scores_list, bbox_preds_list,
                                          batch_gt_instances, batch_img_metas)

        # AlignDETR: `num_total_pos` for matching queries is the number of
        #   unique gt bboxes in the batch. Refer to AlignDETR official code:
        #   https://github.com/FelixCaae/AlignDETR/blob/8c2b1806026e1b33fe1c282577de1647e352d7f0/aligndetr/criterions/base_criterion.py#L195C15-L195C15  # noqa: E501
        num_total_pos = sum(
            len(gt_instances) for gt_instances in batch_gt_instances)

        results = list(results)
        results[-2] = num_total_pos
        return tuple(results)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.
        AlignDETR: This method is based on `DETRHead._get_targets_single`.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            layer_index (int): Decoder layer index for the outputs. Defaults
                to `-1`.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)

        # assigner and sampler
        # AlignDETR: Get `k` of current layer.
        layer_index = img_meta['layer_index']
        num_gt_repeat = self.all_layers_num_gt_repeat[layer_index]
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta,
            k=num_gt_repeat)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # AlignDETR: Get label targets, label weights, and bbox weights.
        target_results = self._get_align_detr_targets_single(
            cls_score,
            bbox_pred,
            gt_labels,
            pos_gt_bboxes,
            pos_inds,
            pos_assigned_gt_inds,
            layer_index,
            is_matching_queries=True)

        label_targets, label_weights, bbox_weights = target_results

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (label_targets, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def _loss_dn_single(self, dn_cls_scores: KeysRecorder,
                        dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.
            AlignDETR: This method is based on `DINOHead._loss_dn_single`.

        Args:
            dn_cls_scores (KeysRecorder): Classification scores of a single
                decoder layer in denoising part, has shape (bs,
                num_denoising_queries, cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        # AlignDETR: Get dn_cls_scores tensor.
        dn_cls_scores = dn_cls_scores.obj

        # AlignDETR: Add layer outputs to meta info because they are not
        #   variables of method `_get_dn_targets_single`.
        for image_index, img_meta in enumerate(batch_img_metas):
            img_meta['dn_cls_score'] = dn_cls_scores[image_index]
            img_meta['dn_bbox_pred'] = dn_bbox_preds[image_index]

        results = super()._loss_dn_single(dn_cls_scores, dn_bbox_preds,
                                          batch_gt_instances, batch_img_metas,
                                          dn_meta)
        return results

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get targets in denoising part for one image.
            AlignDETR: This method is based on
            `DINOHead._get_dn_targets_single`.
            and 1) Added passing `dn_cls_score`, `dn_bbox_pred` to this
            method; 2) Modified the way to get targets.
        Args:
            dn_cls_score (Tensor): Box score logits from a single decoder
            layer in denoising part for one image, has shape
                [num_denoising_queries, cls_out_channels].
            dn_bbox_pred (Tensor): Sigmoid outputs from a single decoder
            layer in denoising part for one image, with
                normalized coordinate (cx, cy, w, h) and shape
                [num_denoising_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # AlignDETR: Get meta info and layer outputs.
        img_h, img_w = img_meta['img_shape']
        dn_cls_score = img_meta['dn_cls_score']
        dn_bbox_pred = img_meta['dn_bbox_pred']
        factor = dn_bbox_pred.new_tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0)

        # AlignDETR: Convert dn_bbox_pred from xywh, normalized to xyxy,
        #   unnormalized.
        dn_bbox_pred = bbox_cxcywh_to_xyxy(dn_bbox_pred)
        dn_bbox_pred = dn_bbox_pred * factor

        # AlignDETR: Get label targets, label weights, and bbox weights.
        target_results = self._get_align_detr_targets_single(
            dn_cls_score, dn_bbox_pred, gt_labels,
            gt_bboxes.repeat([num_groups, 1]), pos_inds, pos_assigned_gt_inds)

        label_targets, label_weights, bbox_weights = target_results

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (label_targets, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def _get_align_detr_targets_single(self,
                                       cls_score: Tensor,
                                       bbox_pred: Tensor,
                                       gt_labels: Tensor,
                                       pos_gt_bboxes: Tensor,
                                       pos_inds: Tensor,
                                       pos_assigned_gt_inds: Tensor,
                                       layer_index: int = -1,
                                       is_matching_queries: bool = False):
        '''AlignDETR: Get label targets, label weights, and bbox weights based
            on `t`, the weighted geometric average of the confident score and
            the IoU score, to align classification and regression scores.

        Args:
            cls_score (Tensor): Box score logits from the last encoder layer
                or a single decoder layer for one image. Shape
                [num_queries or num_denoising_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last encoder layer
                or a single decoder layer for one image, with unnormalized
                coordinate (x, y, x, y) and shape
                [num_queries or num_denoising_queries, 4].
            gt_labels (Tensor): Ground truth classification labels for one
                image, has shape [num_gt].
            pos_gt_bboxes (Tensor): Positive ground truth bboxes for one
                image, with unnormalized coordinate (x, y, x, y) and shape
                [num_positive, 4].
            pos_inds (Tensor): Positive prediction box indices, has shape
                [num_positive].
            pos_assigned_gt_inds Tensor: Positive ground truth box indices,
                has shape [num_positive].
            layer_index (int): decoder layer index for the outputs. Defaults
                to `-1`.
            is_matching_queries (bool): The outputs are from matching
                queries or denoising queries. Defaults to `False`.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - label_targets (Tensor): Labels of one image. Shape
                [num_queries or num_denoising_queries, cls_out_channels].
            - label_weights (Tensor): Label weights of one image. Shape
                [num_queries or num_denoising_queries, cls_out_channels].
            - bbox_weights (Tensor): BBox weights of one image. Shape
                [num_queries or num_denoising_queries, 4].
        '''

        # Classification loss
        # =           1 * BCE(prob, t * rank_weights) for positive sample;
        # = prob**gamma * BCE(prob,                0) for negative sample.
        # That is,
        # label_targets = 0                for negative sample;
        #               = t * rank_weights for positive sample.
        # label_weights = pred**gamma for negative sample;
        #               = 1           for positive sample.
        cls_prob = cls_score.sigmoid()
        label_targets = torch.zeros_like(
            cls_score, device=pos_gt_bboxes.device)
        label_weights = cls_prob**self.gamma

        bbox_weights = torch.zeros_like(bbox_pred, dtype=pos_gt_bboxes.dtype)

        if len(pos_inds) == 0:
            return label_targets, label_weights, bbox_weights

        pos_cls_score_inds = (pos_inds, gt_labels[pos_assigned_gt_inds])
        iou_scores = bbox_overlaps(
            bbox_pred[pos_inds], pos_gt_bboxes, is_aligned=True)

        # t (Tensor): The weighted geometric average of the confident score
        #   and the IoU score, to align classification and regression scores.
        #   Shape [num_positive].
        t = (
            cls_prob[pos_cls_score_inds]**self.alpha *
            iou_scores**(1 - self.alpha))
        t = torch.clamp(t, 0.01).detach()

        # Calculate rank_weights for matching queries.
        if is_matching_queries:
            # rank_weights (Tensor): Weights of each group of predictions
            #   assigned to the same positive gt bbox. Shape [num_positive].
            rank_weights = torch.zeros_like(t, dtype=self.weight_table.dtype)

            assert 0 <= layer_index < len(self.weight_table), layer_index
            rank_to_weight = self.weight_table[layer_index].to(
                rank_weights.device)
            unique_gt_inds = torch.unique(pos_assigned_gt_inds)

            # For each positive gt bbox, get all predictions assigned to it,
            #   then calculate rank weights for this group of predictions.
            for gt_index in unique_gt_inds:
                pred_group_cond = pos_assigned_gt_inds == gt_index
                # Weights are based on their rank sorted by t in the group.
                pred_group = t[pred_group_cond]
                indices = pred_group.sort(descending=True)[1]
                group_weights = torch.zeros_like(
                    indices, dtype=self.weight_table.dtype)
                group_weights[indices] = rank_to_weight[:len(indices)]
                rank_weights[pred_group_cond] = group_weights

            t = t * rank_weights
            pos_bbox_weights = rank_weights.unsqueeze(-1).repeat(
                1, bbox_pred.size(-1))
            bbox_weights[pos_inds] = pos_bbox_weights
        else:
            bbox_weights[pos_inds] = 1.0

        label_targets[pos_cls_score_inds] = t
        label_weights[pos_cls_score_inds] = 1.0

        return label_targets, label_weights, bbox_weights
