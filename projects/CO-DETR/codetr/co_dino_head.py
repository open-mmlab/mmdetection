# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models import DINOHead
from mmdet.models.layers import CdnQueryGenerator
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import InstanceList, reduce_mean


@MODELS.register_module()
class CoDINOHead(DINOHead):

    def __init__(self,
                 *args,
                 num_query=900,
                 transformer=None,
                 in_channels=2048,
                 max_pos_coords=300,
                 dn_cfg=None,
                 use_zero_padding=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 **kwargs):
        self.with_box_refine = True
        self.mixed_selection = True
        self.in_channels = in_channels
        self.max_pos_coords = max_pos_coords
        self.positional_encoding = positional_encoding
        self.num_query = num_query
        self.use_zero_padding = use_zero_padding

        if 'two_stage_num_proposals' in transformer:
            assert transformer['two_stage_num_proposals'] == num_query, \
                'two_stage_num_proposals must be equal to num_query for DINO'
        else:
            transformer['two_stage_num_proposals'] = num_query
        transformer['as_two_stage'] = True
        if self.mixed_selection:
            transformer['mixed_selection'] = self.mixed_selection
        self.transformer = transformer
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))

        super().__init__(*args, **kwargs)

        self.activate = MODELS.build(self.act_cfg)
        self.positional_encoding = MODELS.build(self.positional_encoding)
        self.init_denoising(dn_cfg)

    def _init_layers(self):
        self.transformer = MODELS.build(self.transformer)
        self.embed_dims = self.transformer.embed_dims
        assert hasattr(self.positional_encoding, 'num_feats')
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)

        self.downsample = nn.Sequential(
            nn.Conv2d(
                self.embed_dims,
                self.embed_dims,
                kernel_size=3,
                stride=2,
                padding=1), nn.GroupNorm(32, self.embed_dims))

    def init_denoising(self, dn_cfg):
        if dn_cfg is not None:
            dn_cfg['num_classes'] = self.num_classes
            dn_cfg['num_matching_queries'] = self.num_query
            dn_cfg['embed_dims'] = self.embed_dims
        self.dn_generator = CdnQueryGenerator(**dn_cfg)

    def forward(self,
                mlvl_feats,
                img_metas,
                dn_label_query=None,
                dn_bbox_query=None,
                attn_mask=None):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        hs, inter_references, topk_score, topk_anchor, enc_outputs = \
            self.transformer(
                mlvl_feats,
                mlvl_masks,
                query_embeds,
                mlvl_positional_encodings,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )
        outs = []
        num_level = len(mlvl_feats)
        start = 0
        for lvl in range(num_level):
            bs, c, h, w = mlvl_feats[lvl].shape
            end = start + h * w
            feat = enc_outputs[start:end].permute(1, 2, 0).contiguous()
            start = end
            outs.append(feat.reshape(bs, c, h, w))
        outs.append(self.downsample(outs[-1]))

        hs = hs.permute(0, 2, 1, 3)

        if dn_label_query is not None and dn_label_query.size(1) == 0:
            # NOTE: If there is no target in the image, the parameters of
            # label_embedding won't be used in producing loss, which raises
            # RuntimeError when using distributed mode.
            hs[0] += self.dn_generator.label_embedding.weight[0, 0] * 0.0

        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return outputs_classes, outputs_coords, topk_score, topk_anchor, outs

    def predict(self,
                feats: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self.forward(feats, batch_img_metas)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)

        return predictions

    def predict_by_feat(self,
                        all_cls_scores,
                        all_bbox_preds,
                        enc_cls_scores,
                        enc_bbox_preds,
                        enc_outputs,
                        batch_img_metas,
                        rescale=True):

        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        score_thr = self.test_cfg.get('score_thr', 0)
        with_nms = self.test_cfg.get('nms', None)

        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        if score_thr > 0:
            valid_mask = scores > score_thr
            scores = scores[valid_mask]
            bbox_pred = bbox_pred[valid_mask]
            det_labels = det_labels[valid_mask]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels,
                                                self.test_cfg.nms)
            results = results[keep_idxs]
            results.scores = det_bboxes[:, -1]
            results = results[:max_per_img]

        return results

    def loss(self, x, batch_data_samples):
        assert self.dn_generator is not None, '"dn_cfg" must be set'

        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
            self.dn_generator(batch_data_samples)

        outs = self(x, batch_img_metas, dn_label_query, dn_bbox_query,
                    attn_mask)

        loss_inputs = outs[:-1] + (batch_gt_instances, batch_img_metas,
                                   dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        enc_outputs = outs[-1]
        return losses, enc_outputs

    def forward_aux(self, mlvl_feats, img_metas, aux_targets, head_idx):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        aux_coords, aux_labels, aux_targets, aux_label_weights, \
            aux_bbox_weights, aux_feats, attn_masks = aux_targets
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        hs, inter_references = self.transformer.forward_aux(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            aux_coords,
            pos_feats=aux_feats,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            return_encoder_output=True,
            attn_masks=attn_masks,
            head_idx=head_idx)

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return outputs_classes, outputs_coords, None, None

    def loss_aux(self,
                 x,
                 pos_coords=None,
                 head_idx=0,
                 batch_data_samples=None):
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        gt_bboxes = [b.bboxes for b in batch_gt_instances]
        gt_labels = [b.labels for b in batch_gt_instances]

        aux_targets = self.get_aux_targets(pos_coords, batch_img_metas, x,
                                           head_idx)
        outs = self.forward_aux(x[:-1], batch_img_metas, aux_targets, head_idx)
        outs = outs + aux_targets
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, batch_img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, batch_img_metas)
        losses = self.loss_aux_by_feat(*loss_inputs)
        return losses

    def get_aux_targets(self, pos_coords, img_metas, mlvl_feats, head_idx):
        coords, labels, targets = pos_coords[:3]
        head_name = pos_coords[-1]
        bs, c = len(coords), mlvl_feats[0].shape[1]
        max_num_coords = 0
        all_feats = []
        for i in range(bs):
            label = labels[i]
            feats = [
                feat[i].reshape(c, -1).transpose(1, 0) for feat in mlvl_feats
            ]
            feats = torch.cat(feats, dim=0)
            bg_class_ind = self.num_classes
            pos_inds = ((label >= 0)
                        & (label < bg_class_ind)).nonzero().squeeze(1)
            max_num_coords = max(max_num_coords, len(pos_inds))
            all_feats.append(feats)
        max_num_coords = min(self.max_pos_coords, max_num_coords)
        max_num_coords = max(9, max_num_coords)

        if self.use_zero_padding:
            attn_masks = []
            label_weights = coords[0].new_zeros([bs, max_num_coords])
        else:
            attn_masks = None
            label_weights = coords[0].new_ones([bs, max_num_coords])
        bbox_weights = coords[0].new_zeros([bs, max_num_coords, 4])

        aux_coords, aux_labels, aux_targets, aux_feats = [], [], [], []

        for i in range(bs):
            coord, label, target = coords[i], labels[i], targets[i]
            feats = all_feats[i]
            if 'rcnn' in head_name:
                feats = pos_coords[-2][i]
                num_coords_per_point = 1
            else:
                num_coords_per_point = coord.shape[0] // feats.shape[0]
            feats = feats.unsqueeze(1).repeat(1, num_coords_per_point, 1)
            feats = feats.reshape(feats.shape[0] * num_coords_per_point,
                                  feats.shape[-1])
            img_meta = img_metas[i]
            img_h, img_w = img_meta['img_shape']
            factor = coord.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
            bg_class_ind = self.num_classes
            pos_inds = ((label >= 0)
                        & (label < bg_class_ind)).nonzero().squeeze(1)
            neg_inds = (label == bg_class_ind).nonzero().squeeze(1)
            if pos_inds.shape[0] > max_num_coords:
                indices = torch.randperm(
                    pos_inds.shape[0])[:max_num_coords].cuda()
                pos_inds = pos_inds[indices]

            coord = bbox_xyxy_to_cxcywh(coord[pos_inds] / factor)
            label = label[pos_inds]
            target = bbox_xyxy_to_cxcywh(target[pos_inds] / factor)
            feat = feats[pos_inds]

            if self.use_zero_padding:
                label_weights[i][:len(label)] = 1
                bbox_weights[i][:len(label)] = 1
                attn_mask = torch.zeros([
                    max_num_coords,
                    max_num_coords,
                ]).bool().to(coord.device)
            else:
                bbox_weights[i][:len(label)] = 1

            if coord.shape[0] < max_num_coords:
                padding_shape = max_num_coords - coord.shape[0]
                if self.use_zero_padding:
                    padding_coord = coord.new_zeros([padding_shape, 4])
                    padding_label = label.new_ones([padding_shape
                                                    ]) * self.num_classes
                    padding_target = target.new_zeros([padding_shape, 4])
                    padding_feat = feat.new_zeros([padding_shape, c])
                    attn_mask[coord.shape[0]:, 0:coord.shape[0], ] = True
                    attn_mask[:, coord.shape[0]:, ] = True
                else:
                    indices = torch.randperm(
                        neg_inds.shape[0])[:padding_shape].cuda()
                    neg_inds = neg_inds[indices]
                    padding_coord = bbox_xyxy_to_cxcywh(coords[i][neg_inds] /
                                                        factor)
                    padding_label = labels[i][neg_inds]
                    padding_target = bbox_xyxy_to_cxcywh(targets[i][neg_inds] /
                                                         factor)
                    padding_feat = feats[neg_inds]
                coord = torch.cat((coord, padding_coord), dim=0)
                label = torch.cat((label, padding_label), dim=0)
                target = torch.cat((target, padding_target), dim=0)
                feat = torch.cat((feat, padding_feat), dim=0)
            if self.use_zero_padding:
                attn_masks.append(attn_mask.unsqueeze(0))
            aux_coords.append(coord.unsqueeze(0))
            aux_labels.append(label.unsqueeze(0))
            aux_targets.append(target.unsqueeze(0))
            aux_feats.append(feat.unsqueeze(0))

        if self.use_zero_padding:
            attn_masks = torch.cat(
                attn_masks, dim=0).unsqueeze(1).repeat(1, 8, 1, 1)
            attn_masks = attn_masks.reshape(bs * 8, max_num_coords,
                                            max_num_coords)
        else:
            attn_masks = None

        aux_coords = torch.cat(aux_coords, dim=0)
        aux_labels = torch.cat(aux_labels, dim=0)
        aux_targets = torch.cat(aux_targets, dim=0)
        aux_feats = torch.cat(aux_feats, dim=0)
        aux_label_weights = label_weights
        aux_bbox_weights = bbox_weights
        return (aux_coords, aux_labels, aux_targets, aux_label_weights,
                aux_bbox_weights, aux_feats, attn_masks)

    def loss_aux_by_feat(self,
                         all_cls_scores,
                         all_bbox_preds,
                         enc_cls_scores,
                         enc_bbox_preds,
                         aux_coords,
                         aux_labels,
                         aux_targets,
                         aux_label_weights,
                         aux_bbox_weights,
                         aux_feats,
                         attn_masks,
                         gt_bboxes_list,
                         gt_labels_list,
                         img_metas,
                         gt_bboxes_ignore=None):
        num_dec_layers = len(all_cls_scores)
        all_labels = [aux_labels for _ in range(num_dec_layers)]
        all_label_weights = [aux_label_weights for _ in range(num_dec_layers)]
        all_bbox_targets = [aux_targets for _ in range(num_dec_layers)]
        all_bbox_weights = [aux_bbox_weights for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self._loss_aux_by_feat_single, all_cls_scores, all_bbox_preds,
            all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
            img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.

        # loss from the last decoder layer
        loss_dict['loss_cls_aux'] = losses_cls[-1]
        loss_dict['loss_bbox_aux'] = losses_bbox[-1]
        loss_dict['loss_iou_aux'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_aux'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_aux'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou_aux'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def _loss_aux_by_feat_single(self,
                                 cls_scores,
                                 bbox_preds,
                                 labels,
                                 label_weights,
                                 bbox_targets,
                                 bbox_weights,
                                 img_metas,
                                 gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        num_q = cls_scores.size(1)

        try:
            labels = labels.reshape(num_imgs * num_q)
            label_weights = label_weights.reshape(num_imgs * num_q)
            bbox_targets = bbox_targets.reshape(num_imgs * num_q, 4)
            bbox_weights = bbox_weights.reshape(num_imgs * num_q, 4)
        except Exception:
            return cls_scores.mean() * 0, cls_scores.mean(
            ) * 0, cls_scores.mean() * 0

        bg_class_ind = self.num_classes
        num_total_pos = len(
            ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1))
        num_total_neg = num_imgs * num_q - num_total_pos

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        scores = label_weights.new_zeros(labels.shape)
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
        pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
        pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
        scores[pos_inds] = bbox_overlaps(
            pos_decode_bbox_pred.detach(),
            pos_decode_bbox_targets,
            is_aligned=True)
        loss_cls = self.loss_cls(
            cls_scores, (labels, scores),
            weight=label_weights,
            avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou
