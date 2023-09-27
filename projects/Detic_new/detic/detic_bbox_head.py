# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import List, Optional

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmdet.models.layers import multiclass_nms
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import ConfigType, InstanceList


def load_class_freq(path='datasets/metadata/lvis_v1_train_cat_info.json',
                    freq_weight=0.5):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float()**freq_weight
    return freq_weight


def get_fed_loss_inds(labels, num_sample_cats, C, weight=None):

    appeared = torch.unique(labels)  # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared), replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared


@MODELS.register_module()
class DeticBBoxHead(Shared2FCBBoxHead):

    def __init__(self,
                 image_loss_weight: float = 0.1,
                 use_fed_loss: bool = False,
                 cat_freq_path: str = '',
                 fed_loss_freq_weight: float = 0.5,
                 fed_loss_num_cat: int = 50,
                 cls_predictor_cfg: ConfigType = dict(
                     type='ZeroShotClassifier'),
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reconstruct fc_cls and fc_reg since input channels are changed
        assert self.with_cls

        self.cls_predictor_cfg = cls_predictor_cfg
        cls_channels = self.num_classes
        self.cls_predictor_cfg.update(
            in_features=self.cls_last_dim, out_features=cls_channels)
        self.fc_cls = MODELS.build(self.cls_predictor_cfg)

        self.init_cfg += [
            dict(type='Caffe2Xavier', override=dict(name='reg_fcs'))
        ]

        self.image_loss_weight = image_loss_weight
        self.use_fed_loss = use_fed_loss
        self.cat_freq_path = cat_freq_path
        self.fed_loss_freq_weight = fed_loss_freq_weight
        self.fed_loss_num_cat = fed_loss_num_cat

        if self.use_fed_loss:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        else:
            self.freq_weight = None

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
        scores = cls_score
        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)

        num_classes = 1 if self.reg_class_agnostic else self.num_classes
        roi = roi.repeat_interleave(num_classes, dim=0)
        bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
        bboxes = self.bbox_coder.decode(
            roi[..., 1:], bbox_pred, max_shape=img_shape)

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
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:

            if cls_score.numel() > 0:
                loss_cls_ = self.sigmoid_cross_entropy_loss(cls_score, labels)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]

                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    def sigmoid_cross_entropy_loss(self, cls_score, labels):
        if cls_score.numel() == 0:
            return cls_score.new_zeros(
                [1])[0]  # This is more robust than .sum() * 0.
        B = cls_score.shape[0]
        C = cls_score.shape[1] - 1

        target = cls_score.new_zeros(B, C + 1)
        target[range(len(labels)), labels] = 1  # B x (C + 1)
        target = target[:, :C]  # B x C

        weight = 1
        if self.use_fed_loss and (self.freq_weight is not None):  # fedloss
            appeared = get_fed_loss_inds(
                labels,
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1  # C + 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w.float()
        # if self.ignore_zero_cats and (self.freq_weight is not None):
        #     w = (self.freq_weight.view(-1) > 1e-4).float()
        #     weight = weight * w.view(1, C).expand(B, C)
        #     # import pdb; pdb.set_trace()

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_score[:, :-1], target, reduction='none')  # B x C
        loss = torch.sum(cls_loss * weight) / B
        return loss

    def image_label_losses(self, cls_score, sampling_results, image_labels):
        '''
        Inputs:
            cls_score: N x (C + 1)
            image_labels B x 1
        '''
        num_inst_per_image = [
            len(pred_instances) for pred_instances in sampling_results
        ]
        cls_score = cls_score.split(
            num_inst_per_image, dim=0)  # B x n x (C + 1)
        B = len(cls_score)
        loss = cls_score[0].new_zeros([1])[0]
        for (score, labels, pred_instances) in zip(cls_score, image_labels,
                                                   sampling_results):
            if score.shape[0] == 0:
                loss += score.new_zeros([1])[0]
                continue
            # find out max-size idx
            bboxes = pred_instances.bboxes
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (
                bboxes[:, 3] - bboxes[:, 1])
            idx = areas[:-1].argmax().item() if len(areas) > 1 else 0

            for label in labels:
                target = score.new_zeros(score.shape[1])
                target[label] = 1
                loss_i = F.binary_cross_entropy_with_logits(
                    score[idx], target, reduction='sum')
                loss += loss_i / len(labels)
        loss = loss / B

        return loss * self.image_loss_weight

    def refine_bboxes(self, bbox_results: dict,
                      batch_img_metas: List[dict]) -> InstanceList:
        """Refine bboxes during training.

        Args:
            bbox_results (dict): Usually is a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `rois` (Tensor): RoIs with the shape (n, 5) where the first
                  column indicates batch id of each RoI.
                - `bbox_targets` (tuple):  Ground truth for proposals in a
                  single image. Containing the following list of Tensors:
                  (labels, label_weights, bbox_targets, bbox_weights)
            batch_img_metas (List[dict]): List of image information.

        Returns:
            list[:obj:`InstanceData`]: Refined bboxes of each image.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import numpy as np
            >>> from mmdet.models.task_modules.samplers.
            ... sampling_result import random_boxes
            >>> from mmdet.models.task_modules.samplers import SamplingResult
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            ... batch_img_metas = [{'img_shape': (scale, scale)}
            >>>                     for _ in range(n_img)]
            >>> sampling_results = [SamplingResult.random(rng=10)
            ...                     for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 81, (scale,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> cls_score = torch.randn((scale, 81))
            ... # For each image, pretend random positive boxes are gts
            >>> bbox_targets = (labels, None, None, None)
            ... bbox_results = dict(rois=rois, bbox_pred=bbox_preds,
            ...                     cls_score=cls_score,
            ...                     bbox_targets=bbox_targets)
            >>> bboxes_list = self.refine_bboxes(sampling_results,
            ...                                  bbox_results,
            ...                                  batch_img_metas)
            >>> print(bboxes_list)
        """
        # bbox_targets is a tuple
        cls_scores = bbox_results['cls_score']
        rois = bbox_results['rois']
        bbox_preds = bbox_results['bbox_pred']
        if self.custom_activation:
            # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
            cls_scores = self.loss_cls.get_activation(cls_scores)
        if cls_scores.numel() == 0:
            return None
        if cls_scores.shape[-1] == self.num_classes + 1:
            # remove background class
            cls_scores = cls_scores[:, :-1]
        elif cls_scores.shape[-1] != self.num_classes:
            raise ValueError('The last dim of `cls_scores` should equal to '
                             '`num_classes` or `num_classes + 1`,'
                             f'but got {cls_scores.shape[-1]}.')

        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(batch_img_metas)

        results_list = []
        for i in range(len(batch_img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)

            bboxes_ = rois[inds, 1:]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = batch_img_metas[i]

            bboxes = self.regress(bboxes_, bbox_pred_, img_meta_)

            # don't filter gt bboxes like D2
            results = InstanceData(bboxes=bboxes)
            results_list.append(results)

        return results_list

    def regress(self, priors: Tensor, bbox_pred: Tensor,
                img_meta: dict) -> Tensor:
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            priors (Tensor): Priors from `rpn_head` or last stage
                `bbox_head`, has shape (num_proposals, 4).
            label (Tensor): Only used when `self.reg_class_agnostic`
                is False, has shape (num_proposals, ).
            bbox_pred (Tensor): Regression prediction of
                current stage `bbox_head`. When `self.reg_class_agnostic`
                is False, it has shape (n, num_classes * 4), otherwise
                it has shape (n, 4).
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        reg_dim = self.bbox_coder.encode_size
        assert bbox_pred.size()[1] == reg_dim

        max_shape = img_meta['img_shape']
        regressed_bboxes = self.bbox_coder.decode(
            priors, bbox_pred, max_shape=max_shape)
        return regressed_bboxes
