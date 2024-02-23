# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.runner.amp import autocast
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.layers import multiclass_nms
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, scale_boxes


@MODELS.register_module()
class BaronBBoxHead(BBoxHead):

    def __init__(
            self,
            num_words=4,
            word_dim=512,
            words_drop_ratio=0.5,
            test_cls_temp=None,  # 100.0 / 0.7,
            cls_temp=50.0,
            cls_bias=None,
            cls_embeddings_path=None,
            bg_embedding='zero',
            use_attn12_output=False,
            *args,
            **kwargs):
        super(BaronBBoxHead, self).__init__(*args, **kwargs)
        self.num_words = num_words
        self.word_dim = word_dim
        self.cls_temp = cls_temp
        self.words_drop_ratio = words_drop_ratio
        self.use_attn12_output = use_attn12_output
        assert self.with_cls
        # assert self.reg_class_agnostic
        assert not self.custom_cls_channels
        del self.fc_cls
        self.test_cls_temp = cls_temp if test_cls_temp is None \
            else test_cls_temp

        cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
        cls_predictor_cfg_.update(
            in_features=self.in_channels, out_features=num_words * word_dim)
        self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if cls_bias is None:
            self.cls_bias = 0.0
        else:
            assert self.loss_cls.use_sigmoid, \
                'cls_bias only used for sigmoid logits'
            self.cls_bias = nn.Parameter(torch.ones(1) * cls_bias)
        if cls_embeddings_path is not None:
            cls_embeddings = torch.from_numpy(
                np.load(cls_embeddings_path)).float()
            assert self.num_classes == cls_embeddings.shape[0]
            self.register_buffer('cls_embeddings', cls_embeddings)
            self.learn_bg = False
            if bg_embedding == 'zero':
                self.register_buffer('bg_embedding',
                                     torch.zeros_like(cls_embeddings[:1]))
            elif bg_embedding == 'learn':
                self.bg_embedding = nn.Linear(1, cls_embeddings.shape[1])
                self.init_cfg += [
                    dict(
                        type='Xavier',
                        distribution='uniform',
                        override=dict(name='bg_embedding')),
                ]
                self.learn_bg = True
            else:
                raise ValueError(f'{bg_embedding} not supported.')

    def pred_cls_logits(self, pseudo_words, clip_model):
        text_encoder = clip_model.text_encoder
        if pseudo_words.shape[0] == 0:
            return pseudo_words.new_zeros(0, self.num_classes + 1)
        with autocast():
            valid_mask = self._drop_word(pseudo_words)
            pseudo_text, end_token_ids = \
                text_encoder.prepare_pseudo_text_tensor(
                    pseudo_words, valid_mask)
            # add start and stop token
            if self.use_attn12_output:
                cls_features, _, _ = \
                    text_encoder.encode_pseudo_text_endk(pseudo_text,
                                                         end_token_ids,
                                                         text_pe=True,
                                                         stepk=12,
                                                         normalize=True)
            else:
                cls_features = \
                    text_encoder.encode_pseudo_text(pseudo_text, end_token_ids,
                                                    text_pe=True,
                                                    normalize=True)
        if self.learn_bg:
            input_ones = pseudo_words.new_ones(1, 1)
            bg_embedding = self.bg_embedding(input_ones)
            bg_embedding = F.normalize(bg_embedding, p=2, dim=-1)  # normalize
        else:
            bg_embedding = self.bg_embedding
        cls_embeddings = torch.cat([self.cls_embeddings, bg_embedding])
        if self.training:
            cls_logits = self.cls_temp * cls_features @ cls_embeddings.T
        else:
            cls_logits = self.test_cls_temp * cls_features @ cls_embeddings.T
        if self.training and self.loss_cls.use_sigmoid:
            cls_logits += self.cls_bias
        assert cls_logits.shape[1] == self.num_classes + 1
        return cls_logits

    def _drop_word(self, pseudo_words):
        p = self.words_drop_ratio
        num_preds, num_words, _ = pseudo_words.shape
        mask = F.dropout(
            pseudo_words.new_ones(num_preds, num_words),
            p=p,
            training=self.training)
        start_end_mask = torch.ones_like(mask[:, :1])
        # check empty
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0  # TODO add random on this
        mask[mask > 0.0] = 1.0
        # add start and end token mask
        valid_mask = torch.cat([start_end_mask, mask, start_end_mask], dim=-1)

        return valid_mask

    def forward(self, x, clip_model=None):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        pseudo_words = self.fc_cls(x).view(-1, self.num_words, self.word_dim)
        cls_score = self.pred_cls_logits(pseudo_words, clip_model)
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def vision_to_language(self, x):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        return self.fc_cls(x).view(-1, self.num_words, self.word_dim)

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
        if roi.shape[0] == 0:
            results = InstanceData()
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.loss_cls.use_sigmoid:
            scores = cls_score.sigmoid() if cls_score is not None else None
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        return self._predict_after_normalize_cls_score(roi, scores, bbox_pred,
                                                       img_meta, rescale,
                                                       rcnn_test_cfg)

    def _predict_after_normalize_cls_score(
            self,
            roi: Tensor,
            scores: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None):
        results = InstanceData()
        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
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
