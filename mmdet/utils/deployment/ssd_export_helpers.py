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

from ...core import multiclass_nms
from ...core.bbox.coder.delta_xywh_bbox_coder import delta2bbox
from ...core.anchor.anchor_generator import SSDAnchorGeneratorClustered

from mmdet.utils import add_domain


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
            max_scores, _ = cls_scores[:, :-1].max(dim=1)
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
    def symbolic(g, single_level_grid_anchors, base_anchors, base_size, scales, ratios,
                 anchor_stride, feat, img_tensor, target_stds):
        min_size = base_size
        max_sizes = []
        ars = []
        for scale in scales[1:]:
            max_sizes.append(scale * scale * min_size)
        for ar in ratios:
            if ar > 1:
                ars.append(ar)
        return g.op(add_domain("PriorBox"), feat, img_tensor, min_size_f=[min_size],
                    max_size_f=max_sizes, aspect_ratio_f=ars, flip_i=1,
                    clip_i=0, variance_f=list(target_stds),
                    step_f=anchor_stride[0], offset_f=0.5, step_h_f=0,
                    step_w_f=0, img_size_i=0, img_h_i=0, img_w_i=0)

    @staticmethod
    def forward(ctx, single_level_grid_anchors, base_anchors, base_size, scales, ratios,
                anchor_stride, feat, img_tensor, target_stds):
        assert anchor_stride[0] == anchor_stride[1]
        mlvl_anchor = single_level_grid_anchors(base_anchors, feat.size()[-2:], anchor_stride)
        mlvl_anchor = mlvl_anchor.view(1, -1).unsqueeze(0)
        return mlvl_anchor


class PriorBoxClustered(torch.autograd.Function):
    """Compute priorbox coordinates in point form for each source
    feature map.
    """

    @staticmethod
    def symbolic(g, single_level_grid_anchors, base_anchors, anchors_heights, anchors_widths,
                 anchor_stride, feat, img_tensor, target_stds):
        return g.op(add_domain("PriorBoxClustered"), feat, img_tensor,
                    height_f=anchors_heights, width_f=anchors_widths,
                    flip_i=0, clip_i=0, variance_f=list(target_stds),
                    step_f=anchor_stride[0], offset_f=0.5, step_h_f=0,
                    step_w_f=0, img_size_i=0, img_h_i=0, img_w_i=0)

    @staticmethod
    def forward(ctx, single_level_grid_anchors, base_anchors, anchors_heights, anchors_widths,
                anchor_stride, feat, img_tensor, target_stds):
        assert anchor_stride[0] == anchor_stride[1]
        mlvl_anchor = single_level_grid_anchors(base_anchors, feat.size()[-2:], anchor_stride, base_anchors.device)
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
        return g.op(add_domain("DetectionOutput"), bbox_preds, cls_scores, priors,
                    num_classes_i=cls_out_channels, background_label_id_i=cls_out_channels - 1,
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


def onnx_export(self, img, img_metas, export_name='', **kwargs):
    self._export_mode = True
    self.img_metas = img_metas
    torch.onnx.export(self, img, export_name, **kwargs)


def forward(self, img, img_meta=[None], return_loss=True,
            **kwargs):  # passing None here is a hack to fool the jit engine
    if self._export_mode:
        return self.forward_export(img)
    if return_loss:
        return self.forward_train(img, img_meta, **kwargs)
    else:
        return self.forward_test(img, img_meta, **kwargs)


def forward_export_detector(self, img):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    bbox_result = self.bbox_head.export_forward(*outs, self.test_cfg, True,
                                                self.img_metas, x, img)
    return bbox_result


def export_forward_ssd_head(self, cls_scores, bbox_preds, cfg, rescale,
                            img_metas, feats, img_tensor):
    num_levels = len(cls_scores)

    anchors = []
    for i in range(num_levels):
        if isinstance(self.anchor_generator, SSDAnchorGeneratorClustered):
            anchors.append(PriorBoxClustered.apply(
                self.anchor_generator.single_level_grid_anchors,
                self.anchor_generator.base_anchors[i],
                self.anchor_generator.heights[i],
                self.anchor_generator.widths[i],
                self.anchor_generator.strides[i],
                feats[i], img_tensor, self.bbox_coder.stds))
        else:
            anchors.append(PriorBox.apply(
                self.anchor_generator.single_level_grid_anchors,
                self.anchor_generator.base_anchors[i],
                self.anchor_generator.base_sizes[i],
                self.anchor_generator.scales[i].tolist(),
                self.anchor_generator.ratios[i].tolist(),
                self.anchor_generator.strides[i],
                feats[i],
                img_tensor, self.bbox_coder.stds))
    anchors = torch.cat(anchors, 2)
    cls_scores, bbox_preds = self._prepare_cls_scores_bbox_preds(cls_scores, bbox_preds)

    return DetectionOutput.apply(cls_scores, bbox_preds, img_metas, cfg,
                                 rescale, anchors, self.cls_out_channels,
                                 self.use_sigmoid_cls, self.bbox_coder.means,
                                 self.bbox_coder.stds)


def prepare_cls_scores_bbox_preds_ssd_head(self, cls_scores, bbox_preds):
    scores_list = []
    for o in cls_scores:
        score = o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
        scores_list.append(score)
    cls_scores = torch.cat(scores_list, 1)
    cls_scores = cls_scores.view(cls_scores.size(0), -1, self.cls_out_channels)
    if self.use_sigmoid_cls:
        cls_scores = cls_scores.sigmoid()
    else:
        cls_scores = cls_scores.softmax(-1)
    cls_scores = cls_scores.view(cls_scores.size(0), -1)
    bbox_list = []
    for o in bbox_preds:
        boxes = o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
        bbox_list.append(boxes)
    bbox_preds = torch.cat(bbox_list, 1)
    return cls_scores, bbox_preds
