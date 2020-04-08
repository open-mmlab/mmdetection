# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

# pylint: disable=all

import torch

from ...core import delta2bbox, multiclass_nms


def get_proposals(img_metas, cls_scores, bbox_preds, priors,
                  cfg, rescale, cls_out_channels, use_sigmoid_cls,
                  target_means, target_stds):
    result_list = []
    cls_score_list = cls_scores.tolist()
    bbox_pred_list = bbox_preds.tolist()
    assert len(cls_score_list) == len(bbox_pred_list)
    for img_id in range(len(img_metas)):
        cls_score = \
            torch.Tensor(cls_score_list[img_id]).detach().to(priors.device)
        bbox_pred = \
            torch.Tensor(bbox_pred_list[img_id]).detach().to(priors.device)
        img_shape = img_metas[img_id]['img_shape']
        scale_factor = img_metas[img_id]['scale_factor']
        proposals = get_bboxes_single(cls_score, bbox_pred, priors, img_shape,
                                      scale_factor, cfg, rescale,
                                      cls_out_channels, use_sigmoid_cls,
                                      target_means, target_stds)
        result_list.append(proposals)
    return result_list


def get_bboxes_single(cls_scores, bbox_preds, priors, img_shape, scale_factor,
                      cfg, rescale, cls_out_channels, use_sigmoid_cls,
                      target_means, target_stds):
    cls_scores = cls_scores.view(-1, cls_out_channels)
    bbox_preds = bbox_preds.view(-1, 4)
    priors = priors.view(-1, 4)
    nms_pre = cfg.get('nms_pre', -1)
    if nms_pre > 0 and cls_scores.shape[0] > nms_pre:
        if use_sigmoid_cls:
            max_scores, _ = cls_scores.max(dim=1)
        else:
            max_scores, _ = cls_scores[:, 1:].max(dim=1)
        _, topk_inds = max_scores.topk(nms_pre)
        priors = priors[topk_inds, :]
        bbox_preds = bbox_preds[topk_inds, :]
        cls_scores = cls_scores[topk_inds, :]
    mlvl_bboxes = delta2bbox(priors, bbox_preds, target_means,
                             target_stds, img_shape)
    if rescale:
        mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
    if use_sigmoid_cls:
        padding = cls_scores.new_zeros(cls_scores.shape[0], 1)
        cls_scores = torch.cat([padding, cls_scores], dim=1)
    det_bboxes, det_labels = multiclass_nms(
        mlvl_bboxes, cls_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
    return det_bboxes, det_labels


class PriorBox(torch.autograd.Function):
    """Compute priorbox coordinates in point form for each source
    feature map.
    """

    @staticmethod
    def symbolic(g, anchor_generator, anchor_stride, feat, img_tensor,
                 target_stds):
        min_size = anchor_generator.base_size
        max_sizes = []
        ars = []
        for scale in anchor_generator.scales.tolist()[1:]:
            max_sizes.append(scale * scale * min_size)
        for ar in anchor_generator.ratios.tolist():
            if ar > 1:
                ars.append(ar)
        return g.op("PriorBox", feat, img_tensor, min_size_f=[min_size],
                    max_size_f=max_sizes, aspect_ratio_f=ars, flip_i=1,
                    clip_i=0, variance_f=list(target_stds),
                    step_f=anchor_stride, offset_f=0.5, step_h_f=0,
                    step_w_f=0, img_size_i=0, img_h_i=0, img_w_i=0)

    @staticmethod
    def forward(ctx, anchor_generator, anchor_stride, feat, img_tensor,
                target_stds):

        mlvl_anchor = anchor_generator.grid_anchors(feat.size()[-2:],
                                                    anchor_stride)
        mlvl_anchor = mlvl_anchor.view(1, -1).unsqueeze(0)
        return mlvl_anchor


class PriorBoxClustered(torch.autograd.Function):
    """Compute priorbox coordinates in point form for each source
    feature map.
    """

    @staticmethod
    def symbolic(g, anchor_generator, anchor_stride, feat,
                 img_tensor, target_stds):
        heights = anchor_generator.heights.tolist()
        widths = anchor_generator.widths.tolist()

        return g.op("PriorBoxClustered", feat, img_tensor,
                    height_f=heights, width_f=widths,
                    flip_i=0, clip_i=0, variance_f=list(target_stds),
                    step_f=anchor_stride, offset_f=0.5, step_h_f=0,
                    step_w_f=0, img_size_i=0, img_h_i=0, img_w_i=0)

    @staticmethod
    def forward(ctx, anchor_generator, anchor_stride,
                feat, img_tensor, target_stds):

        mlvl_anchor = anchor_generator.grid_anchors(feat.size()[-2:],
                                                    anchor_stride)
        mlvl_anchor = mlvl_anchor.view(1, -1).unsqueeze(0)
        return mlvl_anchor


class DetectionOutput(torch.autograd.Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    @staticmethod
    def symbolic(g, cls_scores, bbox_preds, img_metas, cfg,
                 rescale, priors, cls_out_channels, use_sigmoid_cls,
                 target_means, target_stds):

        return g.op("DetectionOutput", bbox_preds, cls_scores, priors,
                    num_classes_i=cls_out_channels, background_label_id_i=0,
                    top_k_i=cfg['max_per_img'],
                    keep_top_k_i=cfg['max_per_img'],
                    confidence_threshold_f=cfg['score_thr'],
                    nms_threshold_f=cfg['nms']['iou_thr'],
                    eta_f=1, share_location_i=1,
                    code_type_s="CENTER_SIZE", variance_encoded_in_target_i=0)

    @staticmethod
    def forward(ctx, cls_scores, bbox_preds, img_metas, cfg,
                rescale, priors, cls_out_channels, use_sigmoid_cls,
                target_means, target_stds):

        proposals = get_proposals(img_metas, cls_scores, bbox_preds, priors,
                                  cfg, rescale, cls_out_channels,
                                  use_sigmoid_cls, target_means, target_stds)
        b_s = len(proposals)
        output = \
            torch.zeros(b_s, 1, cfg.max_per_img, 7).to(cls_scores.device)
        for img_id in range(0, b_s):
            bboxes, labels = proposals[img_id]
            coords = bboxes[:, :4]
            scores = bboxes[:, 4]
            labels = labels.float()
            output_for_img = \
                torch.zeros(scores.size()[0], 7).to(cls_scores.device)
            output_for_img[:, 0] = img_id
            output_for_img[:, 1] = labels
            output_for_img[:, 2] = scores
            output_for_img[:, 3:] = coords
            output[img_id, 0, :output_for_img.size()[0]] = output_for_img

        return output
