# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_dn_generator
from mmdet.models.utils.transformer import inverse_sigmoid
from ..builder import HEADS
from .deformable_detr_head import DeformableDETRHead


@HEADS.register_module()
class DINOHead(DeformableDETRHead):

    def __init__(self,
                 *args,
                 num_query=100,
                 dn_cfg=None,
                 transformer=None,
                 **kwargs):

        if 'two_stage_num_proposals' in transformer:
            assert transformer['two_stage_num_proposals'] == num_query, \
                'two_stage_num_proposals must be equal to num_query for DINO'
        else:
            transformer['two_stage_num_proposals'] = num_query
        super(DINOHead, self).__init__(
            *args, num_query=num_query, transformer=transformer, **kwargs)

        assert self.as_two_stage, \
            'as_two_stage must be True for DINO'
        assert self.with_box_refine, \
            'with_box_refine must be True for DINO'
        self._init_layers()
        self.init_denoising(dn_cfg)

    def _init_layers(self):
        super()._init_layers()
        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates [Unknown] class. However, the embedding of unknown class
        # is not used in the original DINO
        self.label_embedding = nn.Embedding(self.cls_out_channels,
                                            self.embed_dims)

    def init_denoising(self, dn_cfg):
        if dn_cfg is not None:
            dn_cfg['num_classes'] = self.num_classes
            dn_cfg['num_queries'] = self.num_query
            dn_cfg['hidden_dim'] = self.embed_dims
        self.dn_generator = build_dn_generator(dn_cfg)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        assert self.dn_generator is not None, '"dn_cfg" must be set'
        dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
            self.dn_generator(gt_bboxes, gt_labels,
                              self.label_embedding, img_metas)
        outs = self(x, img_metas, dn_label_query, dn_bbox_query, attn_mask)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas, dn_meta)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, dn_meta)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

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
            img_h, img_w, _ = img_metas[img_id]['img_shape']
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
        hs, inter_references, topk_score, topk_anchor = \
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
        hs = hs.permute(0, 2, 1, 3)

        if dn_label_query is not None and dn_label_query.size(1) == 0:
            # NOTE: If there is no target in the image, the parameters of
            # label_embedding won't be used in producing loss, which raises
            # RuntimeError when using distributed mode.
            hs[0] += self.label_embedding.weight[0, 0] * 0.0

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

        return outputs_classes, outputs_coords, topk_score, topk_anchor

    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             enc_topk_scores,
             enc_topk_anchors,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             dn_meta=None,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        loss_dict = dict()

        # extract denoising and matching part of outputs
        all_cls_scores, all_bbox_preds, dn_cls_scores, dn_bbox_preds = \
            self.extract_dn_outputs(all_cls_scores, all_bbox_preds, dn_meta)

        if enc_topk_scores is not None:
            # calculate loss from encode feature maps
            # NOTE The DeformDETR calculate binary cls loss
            # for all encoder embeddings, while DINO calculate
            # multi-class loss for topk embeddings.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_topk_scores, enc_topk_anchors,
                                 gt_bboxes_list, gt_labels_list,
                                 img_metas, gt_bboxes_ignore)

            # collate loss from encode feature maps
            loss_dict['interm_loss_cls'] = enc_loss_cls
            loss_dict['interm_loss_bbox'] = enc_losses_bbox
            loss_dict['interm_loss_iou'] = enc_losses_iou

        # calculate loss from all decoder layers
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        # collate loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]

        # collate loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1

        if dn_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_meta = [dn_meta for _ in img_metas]
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                dn_cls_scores, dn_bbox_preds, gt_bboxes_list, gt_labels_list,
                img_metas, dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1],
                    dn_losses_iou[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                num_dec_layer += 1

        return loss_dict

    def loss_dn(self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list,
                gt_labels_list, img_metas, dn_meta):
        num_dec_layers = len(dn_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        dn_meta_list = [dn_meta for _ in range(num_dec_layers)]
        return multi_apply(self.loss_dn_single, dn_cls_scores, dn_bbox_preds,
                           all_gt_bboxes_list, all_gt_labels_list,
                           img_metas_list, dn_meta_list)

    def loss_dn_single(self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list,
                       gt_labels_list, img_metas, dn_meta):
        num_imgs = dn_cls_scores.size(0)
        bbox_preds_list = [dn_bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_dn_target(bbox_preds_list, gt_bboxes_list,
                                             gt_labels_list, img_metas,
                                             dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(  # TODO: How to better return zero loss
                1,
                dtype=cls_scores.dtype,
                device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, dn_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def get_dn_target(self, dn_bbox_preds_list, gt_bboxes_list, gt_labels_list,
                      img_metas, dn_meta):
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_dn_target_single,
                                      dn_bbox_preds_list, gt_bboxes_list,
                                      gt_labels_list, img_metas, dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_dn_target_single(self, dn_bbox_pred, gt_bboxes, gt_labels,
                              img_meta, dn_meta):
        num_groups = dn_meta['num_dn_group']
        pad_size = dn_meta['pad_size']
        assert pad_size % num_groups == 0
        single_pad = pad_size // num_groups
        num_bboxes = dn_bbox_pred.size(0)

        if len(gt_labels) > 0:
            t = torch.range(0, len(gt_labels) - 1).long().cuda()
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = (torch.tensor(range(num_groups)) *
                        single_pad).long().cuda().unsqueeze(1) + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = torch.tensor([]).long().cuda()
        neg_inds = pos_inds + single_pad // 2

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(dn_bbox_pred)
        bbox_weights = torch.zeros_like(dn_bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = dn_bbox_pred.new_tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    @staticmethod
    def extract_dn_outputs(all_cls_scores, all_bbox_preds, dn_meta):
        if dn_meta is not None:
            denoising_cls_scores = all_cls_scores[:, :, :
                                                  dn_meta['pad_size'], :]
            denoising_bbox_preds = all_bbox_preds[:, :, :
                                                  dn_meta['pad_size'], :]
            matching_cls_scores = all_cls_scores[:, :, dn_meta['pad_size']:, :]
            matching_bbox_preds = all_bbox_preds[:, :, dn_meta['pad_size']:, :]
        else:
            denoising_cls_scores = None
            denoising_bbox_preds = None
            matching_cls_scores = all_cls_scores
            matching_bbox_preds = all_bbox_preds
        return (matching_cls_scores, matching_bbox_preds, denoising_cls_scores,
                denoising_bbox_preds)
