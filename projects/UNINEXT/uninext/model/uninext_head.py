# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from mmcv.cnn import Linear
from mmengine.model import BaseModule, bias_init_with_prob, constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.layers import inverse_sigmoid
from mmdet.models.layers.transformer import MLP
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import ConfigType, InstanceList
from .layers import VL_Align
from .mask_head import (MaskHeadSmallConv, aligned_bilinear, compute_locations,
                        parse_dynamic_params)
from .utils import convert_grounding_to_od_logits


@MODELS.register_module()
class UNINEXTHead(BaseModule):
    r"""Head of the DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2203.03605>`_ .
    """

    def __init__(self,
                 embed_dims: int = 256,
                 language_dims: int = 768,
                 repeat_nums: int = 2,
                 add_iou_branch: bool = True,
                 as_two_stage: bool = True,
                 share_pred_layer: bool = False,
                 num_pred_layer: int = 6,
                 use_rel_coord=True,
                 use_raft=False,
                 inference_select_thres: float = 0.1,
                 reid_head: ConfigType = dict(
                     type='DeformableReidHead',
                     num_layers=2,
                     layer_cfg=dict(
                         self_attn_cfg=dict(
                             embed_dims=256, num_heads=8, dropout=0.0),
                         cross_attn_cfg=dict(
                             embed_dims=256, num_levels=4, dropout=0.0),
                         ffn_cfg=dict(
                             embed_dims=256,
                             feedforward_channels=2048,
                             ffn_drop=0.0))),
                 **kwargs) -> None:
        super(UNINEXTHead, self).__init__()
        self.embed_dims = embed_dims
        self.language_dims = language_dims
        self.rel_coord = use_rel_coord
        self.inference_select_thres = inference_select_thres
        self.use_raft = use_raft
        self.reid_head = reid_head
        self.repeat_nums = repeat_nums
        self.add_iou_branch = add_iou_branch
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage
        self.num_feature_levels = 4
        self.new_mask_head = False

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""

        # dynamic_mask_head params
        self.in_channels = self.embed_dims // 32
        self.dynamic_mask_channels = 8
        self.controller_layers = 3
        self.max_insts_num = 100
        self.mask_out_stride = 4
        self.up_rate = 8 // self.mask_out_stride

        weight_nums, bias_nums = [], []
        for index in range(self.controller_layers):
            if index == 0:
                if self.rel_coord:
                    weight_nums.append(
                        (self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels *
                                       self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif index == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels *
                                   self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.controller = MLP(self.embed_dims, self.embed_dims,
                              self.num_gen_params, 3)
        self.mask_head = MaskHeadSmallConv(self.embed_dims, None,
                                           self.embed_dims, self.use_raft,
                                           self.up_rate)

        self.reid_branch = MODELS.build(self.reid_head)
        still_classifier = Linear(self.embed_dims, 1)

        if self.add_iou_branch:
            fc_iou = Linear(self.embed_dims, 1)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList([
                VL_Align(self.language_dims, self.embed_dims)
                for _ in range(self.num_pred_layer - 1)
            ])
            self.cls_branches.append(still_classifier)
            self.iou_branches = nn.ModuleList(
                [fc_iou for _ in range(self.num_pred_layer - 1)])
            self.reg_branches = nn.ModuleList([
                MLP(self.embed_dims, self.embed_dims, 4, 3)
                for _ in range(self.num_pred_layer)
            ])
        else:
            self.cls_branches = nn.ModuleList([
                copy.deepcopy(VL_Align(self.language_dims, self.embed_dims))
                for _ in range(self.num_pred_layer - 1)
            ])
            self.cls_branches.append(still_classifier)
            self.iou_branches = nn.ModuleList([
                copy.deepcopy(fc_iou) for _ in range(self.num_pred_layer - 1)
            ])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(MLP(self.embed_dims, self.embed_dims, 4, 3))
                for _ in range(self.num_pred_layer)
            ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branches[-1].bias, bias_init)
        for m in self.iou_branches:
            nn.init.constant_(m.bias, bias_init)

        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
        for contr in self.controller.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)

    def forward_mask_head(self, feats, spatial_shapes, reference_points,
                          mask_head_params, num_insts):
        bs, _, c = feats.shape

        encod_feat_l = []
        spatial_indx = 0
        for feat_l in range(self.num_feature_levels - 1):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:, spatial_indx:spatial_indx + 1 * h * w, :].reshape(
                bs, 1, h, w, c).permute(0, 4, 1, 2, 3)
            encod_feat_l.append(mem_l)
            spatial_indx += 1 * h * w

        pred_masks = []
        for iframe in range(1):
            encod_feat_f = []
            for lvl in range(self.num_feature_levels - 1):
                encod_feat_f.append(
                    encod_feat_l[lvl][:, :, iframe, :, :])  # [bs, C, hi, wi]

            if self.new_mask_head:
                if self.use_raft:
                    decod_feat_f, up_masks = self.mask_head(encod_feat_f)
                else:
                    decod_feat_f = self.mask_head(encod_feat_f)
                    up_masks = None
            else:
                if self.use_raft:
                    decod_feat_f, up_masks = self.mask_head(
                        encod_feat_f, fpns=None)
                else:
                    decod_feat_f = self.mask_head(encod_feat_f, fpns=None)
                    up_masks = None
            mask_logits = self.dynamic_mask_with_coords(
                decod_feat_f,
                reference_points,
                mask_head_params,
                num_insts=num_insts,
                mask_feat_stride=8,
                rel_coord=self.rel_coord,
                up_masks=up_masks)
            # mask_logits: [1, num_queries_all, H/4, W/4]

            mask_f = []
            inst_st = 0
            for num_inst in num_insts:
                # [1, selected_queries, 1, H/4, W/4]
                mask_f.append(mask_logits[:, inst_st:inst_st +
                                          num_inst, :, :].unsqueeze(2))
                inst_st += num_inst

            pred_masks.append(mask_f)

        output_pred_masks = []
        for i, num_inst in enumerate(num_insts):
            out_masks_b = [m[i] for m in pred_masks]
            output_pred_masks.append(torch.cat(out_masks_b, dim=2))

        return output_pred_masks

    def dynamic_mask_with_coords(self,
                                 mask_feats,
                                 reference_points,
                                 mask_head_params,
                                 num_insts,
                                 mask_feat_stride,
                                 rel_coord=True,
                                 up_masks=None):
        # mask_feats: [N, C/32, H/8, W/8]
        # reference_points: [1, \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        # return:
        #     mask_logits: [1, \sum{num_queries}, H/8, W/8]
        device = mask_feats.device

        N, in_channels, H, W = mask_feats.size()
        num_insts_all = reference_points.shape[1]

        locations = compute_locations(
            mask_feats.size(2),
            mask_feats.size(3),
            device=device,
            stride=mask_feat_stride)
        # locations: [H*W, 2]

        if rel_coord:
            instance_locations = reference_points
            relative_coords = instance_locations.reshape(
                1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)

            relative_coords = relative_coords.float()
            relative_coords = relative_coords.permute(0, 1, 4, 2,
                                                      3).flatten(-2, -1)
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                relative_coords_b = relative_coords[:, inst_st:inst_st +
                                                    num_inst, :, :]
                mask_feats_b = mask_feats[i].reshape(
                    1, in_channels,
                    H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = torch.cat([relative_coords_b, mask_feats_b],
                                        dim=2)

                mask_head_inputs.append(mask_head_b)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels,
                                                    H * W).unsqueeze(1).repeat(
                                                        1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)

        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)

        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(mask_head_params,
                                                   self.dynamic_mask_channels,
                                                   self.weight_nums,
                                                   self.bias_nums)

            mask_logits = self.mask_heads_forward(mask_head_inputs, weights,
                                                  biases,
                                                  mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs + torch.sum(mask_head_params) * 0.0
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        mask_logits = mask_logits.reshape(-1, 1, H, W)

        # upsample predicted masks
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0

        if self.use_raft:
            assert up_masks is not None
            inst_idx = 0
            mask_logits_output = []
            for b, n in enumerate(num_insts):
                mask_logits_output.append(
                    self.upsample_preds(mask_logits[inst_idx:inst_idx + n],
                                        up_masks[b:b + 1]))
                inst_idx += n
            mask_logits = torch.cat(mask_logits_output, dim=0)
        else:
            mask_logits = aligned_bilinear(
                mask_logits, int(mask_feat_stride / self.mask_out_stride))

        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2],
                                          mask_logits.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def upsample_preds(self, pred, mask):
        """Upsample pred [N, 1, H/8, W/8] -> [N, 1, H, W] using convex
        combination."""
        N, _, H, W = pred.shape
        mask = mask.view(1, 1, 9, self.up_rate, self.up_rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_pred = F.unfold(pred, [3, 3], padding=1)
        up_pred = up_pred.view(N, 1, 9, 1, 1, H, W)

        up_pred = torch.sum(mask * up_pred, dim=2)
        up_pred = up_pred.permute(0, 1, 4, 2, 5, 3)
        return up_pred.reshape(N, 1, self.up_rate * H, self.up_rate * W)

    def predict(self,
                memory,
                language_dict_features,
                hidden_states,
                inter_references,
                src_info_dict,
                positive_map_label_to_token,
                num_classes,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self.inference_forward(memory, language_dict_features,
                                      hidden_states, inter_references,
                                      src_info_dict, batch_img_metas)

        predictions = self.predict_by_feat(
            *outs,
            positive_map_label_to_token,
            num_classes,
            batch_img_metas=batch_img_metas,
            rescale=rescale)

        return predictions

    def inference_forward(self, memory, language_dict_features, hidden_states,
                          inter_references, src_info_dict,
                          batch_img_metas: List[dict]) -> Tuple[Tensor, ...]:

        src_info_dict['reference_points'] = inter_references[-1].detach()

        reference = inter_references[-2]
        reference = inverse_sigmoid(reference)
        outputs_class = self.cls_branches[-2](hidden_states[-1],
                                              language_dict_features['hidden'])
        tmp = self.reg_branches[-2](hidden_states[-1])
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()

        outputs_embeds = self.reid_branch(
            hidden_states[-1],
            src_info_dict['reference_points'],
            src_info_dict['src'],
            src_info_dict['src_spatial_shapes'],
            src_info_dict['src_level_start_index'],
            src_info_dict['src_valid_ratios'],
        )

        dynamic_mask_head_params = self.controller(
            hidden_states[-1])  # [bs, num_queries, num_params]
        norm_reference_points = inter_references[-2, :, :, :2]
        bs, num_queries, _ = dynamic_mask_head_params.shape
        num_insts = [num_queries for _ in range(bs)]
        reference_points = []
        for batch_id in range(bs):
            img_h, img_w = batch_img_metas[batch_id]['img_shape']
            img_h = torch.as_tensor(img_h).to(norm_reference_points[batch_id])
            img_w = torch.as_tensor(img_w).to(norm_reference_points[batch_id])
            scale_f = torch.stack([img_w, img_h], dim=0)
            ref_cur_f = norm_reference_points[batch_id] * scale_f[None, :]
            reference_points.append(ref_cur_f.unsqueeze(0))

        reference_points = torch.cat(reference_points, dim=1)
        mask_head_params = dynamic_mask_head_params.reshape(
            1, -1, dynamic_mask_head_params.shape[-1])

        spatial_shapes = src_info_dict['src_spatial_shapes']
        outputs_masks = self.forward_mask_head(memory, spatial_shapes,
                                               reference_points,
                                               mask_head_params, num_insts)
        outputs_masks = torch.cat(outputs_masks, dim=0)
        # not support two_stage yet
        return outputs_class, outputs_coord, \
            outputs_masks, outputs_embeds

    def predict_by_feat(self,
                        all_cls_scores: Tensor,
                        all_bbox_preds: Tensor,
                        all_mask_preds: Tensor,
                        all_embeds_preds: Tensor,
                        positive_map_label_to_token,
                        num_classes,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:

        result_list = []
        for img_id in range(len(batch_img_metas)):
            logits = all_cls_scores[img_id]
            bbox_pred = all_bbox_preds[img_id]
            mask_pred = all_mask_preds[img_id]
            embed_pred = all_embeds_preds[img_id]
            img_meta = batch_img_metas[img_id]
            logits = convert_grounding_to_od_logits(
                logits.unsqueeze(0), num_classes, positive_map_label_to_token)
            logits = logits[0]
            results = self._predict_by_feat_single(logits, bbox_pred,
                                                   mask_pred, embed_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                logits: Tensor,
                                bbox_pred: Tensor,
                                mask_pred: Tensor,
                                embed_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:

        img_shape = img_meta['img_shape']
        scores = logits.sigmoid().cpu().detach()
        max_score, _ = torch.max(logits.sigmoid(), 1)
        indices = torch.nonzero(
            max_score > self.inference_select_thres, as_tuple=False).squeeze(1)
        if len(indices) == 0:
            topkv, indices_top1 = torch.topk(scores.max(1)[0], k=1)
            indices_top1 = indices_top1[torch.argmax(topkv)]
            indices = [indices_top1.tolist()]
        else:
            nms_scores, idxs = torch.max(logits.sigmoid()[indices], 1)
            boxes_before_nms = bbox_cxcywh_to_xyxy(bbox_pred[indices])
            keep_indices = ops.batched_nms(boxes_before_nms, nms_scores, idxs,
                                           0.9)
            indices = indices[keep_indices]

        box_score = torch.max(logits.sigmoid()[indices], 1)[0]
        det_labels = torch.argmax(logits.sigmoid()[indices], dim=1)
        track_feats = embed_pred[indices]
        det_masks = mask_pred[indices]
        det_bboxes = bbox_pred[indices]
        scores = scores[indices]

        if rescale:
            det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
            det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
            det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
            det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        det_bboxes = torch.cat([det_bboxes, box_score.unsqueeze(1)], dim=1)

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        results.masks = det_masks
        results.track_feats = track_feats
        return results
