# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmengine.config import ConfigDict
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from torch import Tensor

try:
    from transformers import BertConfig
except ImportError:
    BertConfig = None

from mmdet.registry import MODELS
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from ..utils import (BertEncoderLayer, VLFuse, filter_scores_and_topk,
                     permute_and_flatten, select_single_mlvl,
                     unpack_gt_instances)
from ..utils.vlfuse_helper import MAX_CLAMP_VALUE
from .atss_head import ATSSHead


def convert_grounding_to_cls_scores(logits: Tensor,
                                    positive_maps: List[dict]) -> Tensor:
    """Convert logits to class scores."""
    assert len(positive_maps) == logits.shape[0]  # batch size

    scores = torch.zeros(logits.shape[0], logits.shape[1],
                         len(positive_maps[0])).to(logits.device)
    if positive_maps is not None:
        if all(x == positive_maps[0] for x in positive_maps):
            # only need to compute once
            positive_map = positive_maps[0]
            for label_j in positive_map:
                scores[:, :, label_j -
                       1] = logits[:, :,
                                   torch.LongTensor(positive_map[label_j]
                                                    )].mean(-1)
        else:
            for i, positive_map in enumerate(positive_maps):
                for label_j in positive_map:
                    scores[i, :, label_j - 1] = logits[
                        i, :, torch.LongTensor(positive_map[label_j])].mean(-1)
    return scores


class Conv3x3Norm(nn.Module):
    """Conv3x3 and norm."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 groups: int = 1,
                 use_dcn: bool = False,
                 norm_type: Optional[Union[Sequence, str]] = None):
        super().__init__()

        if use_dcn:
            self.conv = ModulatedDeformConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups)

        if isinstance(norm_type, Sequence):
            assert len(norm_type) == 2
            assert norm_type[0] == 'gn'
            gn_group = norm_type[1]
            norm_type = norm_type[0]

        if norm_type == 'bn':
            bn_op = nn.BatchNorm2d(out_channels)
        elif norm_type == 'gn':
            bn_op = nn.GroupNorm(
                num_groups=gn_group, num_channels=out_channels)
        if norm_type is not None:
            self.bn = bn_op
        else:
            self.bn = None

    def forward(self, x, **kwargs):
        x = self.conv(x, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class DyReLU(nn.Module):
    """Dynamic ReLU."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.expand_ratio = expand_ratio
        self.out_channels = out_channels

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // expand_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // expand_ratio,
                      out_channels * self.expand_ratio),
            nn.Hardsigmoid(inplace=True))

    def forward(self, x) -> Tensor:
        x_out = x
        b, c, h, w = x.size()
        x = self.avg_pool(x).view(b, c)
        x = self.fc(x).view(b, -1, 1, 1)

        a1, b1, a2, b2 = torch.split(x, self.out_channels, dim=1)
        a1 = (a1 - 0.5) * 2 + 1.0
        a2 = (a2 - 0.5) * 2
        b1 = b1 - 0.5
        b2 = b2 - 0.5
        out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        return out


class DyConv(nn.Module):
    """Dynamic Convolution."""

    def __init__(self,
                 conv_func: Callable,
                 in_channels: int,
                 out_channels: int,
                 use_dyfuse: bool = True,
                 use_dyrelu: bool = False,
                 use_dcn: bool = False):
        super().__init__()

        self.dyconvs = nn.ModuleList()
        self.dyconvs.append(conv_func(in_channels, out_channels, 1))
        self.dyconvs.append(conv_func(in_channels, out_channels, 1))
        self.dyconvs.append(conv_func(in_channels, out_channels, 2))

        if use_dyfuse:
            self.attnconv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.ReLU(inplace=True))
            self.h_sigmoid = nn.Hardsigmoid(inplace=True)
        else:
            self.attnconv = None

        if use_dyrelu:
            self.relu = DyReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()

        if use_dcn:
            self.offset = nn.Conv2d(
                in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None

        self.init_weights()

    def init_weights(self):
        for m in self.dyconvs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.attnconv is not None:
            for m in self.attnconv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs: dict) -> dict:
        visual_feats = inputs['visual']

        out_vis_feats = []
        for level, feature in enumerate(visual_feats):

            offset_conv_args = {}
            if self.offset is not None:
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                offset_conv_args = dict(offset=offset, mask=mask)

            temp_feats = [self.dyconvs[1](feature, **offset_conv_args)]

            if level > 0:
                temp_feats.append(self.dyconvs[2](visual_feats[level - 1],
                                                  **offset_conv_args))
            if level < len(visual_feats) - 1:
                temp_feats.append(
                    F.upsample_bilinear(
                        self.dyconvs[0](visual_feats[level + 1],
                                        **offset_conv_args),
                        size=[feature.size(2),
                              feature.size(3)]))
            mean_feats = torch.mean(
                torch.stack(temp_feats), dim=0, keepdim=False)

            if self.attnconv is not None:
                attn_feat = []
                res_feat = []
                for feat in temp_feats:
                    res_feat.append(feat)
                    attn_feat.append(self.attnconv(feat))

                res_feat = torch.stack(res_feat)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_feat))

                mean_feats = torch.mean(
                    res_feat * spa_pyr_attn, dim=0, keepdim=False)

            out_vis_feats.append(mean_feats)

        out_vis_feats = [self.relu(item) for item in out_vis_feats]

        features_dict = {'visual': out_vis_feats, 'lang': inputs['lang']}

        return features_dict


class VLFusionModule(BaseModel):
    """Visual-lang Fusion Module."""

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 num_base_priors: int,
                 early_fuse: bool = False,
                 num_dyhead_blocks: int = 6,
                 lang_model_name: str = 'bert-base-uncased',
                 use_dyrelu: bool = True,
                 use_dyfuse: bool = True,
                 use_dcn: bool = True,
                 use_checkpoint: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if BertConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_base_priors = num_base_priors
        self.early_fuse = early_fuse
        self.num_dyhead_blocks = num_dyhead_blocks
        self.use_dyrelu = use_dyrelu
        self.use_dyfuse = use_dyfuse
        self.use_dcn = use_dcn
        self.use_checkpoint = use_checkpoint

        self.lang_cfg = BertConfig.from_pretrained(lang_model_name)
        self.lang_dim = self.lang_cfg.hidden_size
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the model."""
        bias_value = -math.log((1 - 0.01) / 0.01)

        dyhead_tower = []
        for i in range(self.num_dyhead_blocks):
            if self.early_fuse:
                # cross-modality fusion
                dyhead_tower.append(VLFuse(use_checkpoint=self.use_checkpoint))
                # lang branch
                dyhead_tower.append(
                    BertEncoderLayer(
                        self.lang_cfg,
                        clamp_min_for_underflow=True,
                        clamp_max_for_overflow=True))

            # vision branch
            dyhead_tower.append(
                DyConv(
                    lambda i, o, s: Conv3x3Norm(
                        i, o, s, use_dcn=self.use_dcn, norm_type=['gn', 16]),
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    use_dyrelu=(self.use_dyrelu
                                and self.in_channels == self.feat_channels)
                    if i == 0 else self.use_dyrelu,
                    use_dyfuse=(self.use_dyfuse
                                and self.in_channels == self.feat_channels)
                    if i == 0 else self.use_dyfuse,
                    use_dcn=(self.use_dcn
                             and self.in_channels == self.feat_channels)
                    if i == 0 else self.use_dcn,
                ))

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        self.bbox_pred = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, kernel_size=1)
        self.centerness = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 1, kernel_size=1)
        self.dot_product_projection_text = nn.Linear(
            self.lang_dim,
            self.num_base_priors * self.feat_channels,
            bias=True)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.bias_lang = nn.Parameter(
            torch.zeros(self.lang_dim), requires_grad=True)
        self.bias0 = nn.Parameter(
            torch.Tensor([bias_value]), requires_grad=True)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self, visual_feats: Tuple[Tensor],
                language_feats: dict) -> Tuple:
        feat_inputs = {'visual': visual_feats, 'lang': language_feats}
        dyhead_tower = self.dyhead_tower(feat_inputs)

        if self.early_fuse:
            embedding = dyhead_tower['lang']['hidden']
        else:
            embedding = language_feats['embedded']

        embedding = F.normalize(embedding, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding /
                                                                   2.0)
        dot_product_proj_tokens_bias = torch.matmul(
            embedding, self.bias_lang) + self.bias0

        bbox_preds = []
        centerness = []
        cls_logits = []

        for i, feature in enumerate(visual_feats):
            visual = dyhead_tower['visual'][i]
            B, C, H, W = visual.shape

            bbox_pred = self.scales[i](self.bbox_pred(visual))
            bbox_preds.append(bbox_pred)
            centerness.append(self.centerness(visual))

            dot_product_proj_queries = permute_and_flatten(
                visual, B, self.num_base_priors, C, H, W)

            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(
                1, self.num_base_priors, 1)
            dot_product_logit = (
                torch.matmul(dot_product_proj_queries,
                             dot_product_proj_tokens.transpose(-1, -2)) /
                self.log_scale.exp()) + bias
            dot_product_logit = torch.clamp(
                dot_product_logit, max=MAX_CLAMP_VALUE)
            dot_product_logit = torch.clamp(
                dot_product_logit, min=-MAX_CLAMP_VALUE)
            cls_logits.append(dot_product_logit)

        return bbox_preds, centerness, cls_logits


@MODELS.register_module()
class ATSSVLFusionHead(ATSSHead):
    """ATSS head with visual-language fusion module.

    Args:
        early_fuse (bool): Whether to fuse visual and language features
            Defaults to False.
        use_checkpoint (bool): Whether to use checkpoint. Defaults to False.
        num_dyhead_blocks (int): Number of dynamic head blocks. Defaults to 6.
        lang_model_name (str): Name of the language model.
            Defaults to 'bert-base-uncased'.
    """

    def __init__(self,
                 *args,
                 early_fuse: bool = False,
                 use_checkpoint: bool = False,
                 num_dyhead_blocks: int = 6,
                 lang_model_name: str = 'bert-base-uncased',
                 init_cfg=None,
                 **kwargs):
        super().__init__(*args, **kwargs, init_cfg=init_cfg)
        self.head = VLFusionModule(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            num_base_priors=self.num_base_priors,
            early_fuse=early_fuse,
            use_checkpoint=use_checkpoint,
            num_dyhead_blocks=num_dyhead_blocks,
            lang_model_name=lang_model_name)
        self.text_masks = None

    def _init_layers(self) -> None:
        """No need to initialize the ATSS head layer."""
        pass

    def forward(self, visual_feats: Tuple[Tensor],
                language_feats: dict) -> Tuple[Tensor]:
        """Forward function."""
        bbox_preds, centerness, cls_logits = self.head(visual_feats,
                                                       language_feats)
        return cls_logits, bbox_preds, centerness

    def loss(self, visual_feats: Tuple[Tensor], language_feats: dict,
             batch_data_samples):
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(visual_feats, language_feats)
        self.text_masks = language_feats['masks']
        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            centernesses: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets
        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        anchors = torch.cat(anchor_list, dim=1)
        labels = torch.cat(labels_list, dim=1)
        label_weights = torch.cat(label_weights_list, dim=1)
        bbox_targets = torch.cat(bbox_targets_list, dim=1)
        cls_scores = torch.cat(cls_scores, dim=1)

        centernesses_ = []
        bbox_preds_ = []
        for bbox_pred, centerness in zip(bbox_preds, centernesses):
            centernesses_.append(
                centerness.permute(0, 2, 3,
                                   1).reshape(cls_scores.size(0), -1, 1))
            bbox_preds_.append(
                bbox_pred.permute(0, 2, 3,
                                  1).reshape(cls_scores.size(0), -1, 4))
        bbox_preds = torch.cat(bbox_preds_, dim=1)
        centernesses = torch.cat(centernesses_, dim=1)

        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor = \
            self._loss_by_feat(
                anchors,
                cls_scores,
                bbox_preds,
                centernesses,
                labels,
                label_weights,
                bbox_targets,
                avg_factor=avg_factor)

        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = losses_bbox / bbox_avg_factor
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness)

    def _loss_by_feat(self, anchors: Tensor, cls_score: Tensor,
                      bbox_pred: Tensor, centerness: Tensor, labels: Tensor,
                      label_weights: Tensor, bbox_targets: Tensor,
                      avg_factor: float) -> dict:
        """Calculate the loss of all scale level based on the features
        extracted by the detection head.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)

        # ===== this change =====
        pos_inds = (labels.sum(-1) > 0).reshape(-1)

        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_mask = (self.text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_score.size(1), 1)
        cls_score = torch.masked_select(cls_score, text_mask).contiguous()
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        bbox_pred = bbox_pred.reshape(-1, 4)
        centerness = centerness.reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)

        if pos_inds.sum() > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)

            if torch.isnan(centerness_targets).any():
                print('=====Centerness includes NaN=====')
                mask = ~torch.isnan(centerness_targets)
                centerness_targets = centerness_targets[mask]
                pos_centerness = pos_centerness[mask]
                pos_anchors = pos_anchors[mask]
                pos_bbox_targets = pos_bbox_targets[mask]
                pos_bbox_pred = pos_bbox_pred[mask]

                if pos_bbox_targets.shape[0] == 0:
                    loss_bbox = bbox_pred.sum() * 0
                    loss_centerness = centerness.sum() * 0
                    centerness_targets = bbox_targets.new_tensor(0.)
                    return loss_cls, loss_bbox, loss_centerness, \
                        centerness_targets.sum()

            # The decoding process takes the offset into consideration.
            pos_anchors[:, 2:] += 1
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness, centerness_targets, avg_factor=avg_factor)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def _get_targets_single(self,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            num_level_anchors: List[int],
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors (List[int]): Number of anchors of each scale
                level.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
                sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        anchors = flat_anchors
        # Align the official implementation
        anchors[:, 2:] -= 1

        num_level_anchors_inside = num_level_anchors
        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances,
                                             num_level_anchors_inside,
                                             gt_instances, gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)

        # ===== this change =====
        labels = anchors.new_full((num_valid_anchors, self.feat_channels),
                                  0,
                                  dtype=torch.float32)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            # ===== this change =====
            labels[pos_inds] = gt_instances.positive_maps[
                sampling_result.pos_assigned_gt_inds]
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, sampling_result)

    def centerness_target(self, anchors: Tensor, gts: Tensor) -> Tensor:
        """Calculate the centerness between anchors and gts.

        Only calculate pos centerness targets, otherwise there may be nan.

        Args:
            anchors (Tensor): Anchors with shape (N, 4), "xyxy" format.
            gts (Tensor): Ground truth bboxes with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Centerness between anchors and gts.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        # assert not torch.isnan(centerness).any()
        return centerness

    def predict(self,
                visual_feats: Tuple[Tensor],
                language_feats: dict,
                batch_data_samples,
                rescale: bool = True):
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            visual_feats (tuple[Tensor]): Multi-level visual features from the
                upstream network, each is a 4D-tensor.
            language_feats (dict): Language features from the upstream network.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]
        outs = self(visual_feats, language_feats)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        cls_logits: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_logits (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            batch_token_positive_maps (list[dict], Optional): Batch token
                positive map. Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(bbox_preds) == len(score_factors)
        num_levels = len(bbox_preds)

        featmap_sizes = [bbox_preds[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            token_positive_maps = batch_token_positive_maps[img_id]
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            score_factor_list = select_single_mlvl(
                score_factors, img_id, detach=True)
            cls_logit_list = select_single_mlvl(
                cls_logits, img_id, detach=True)

            results = self._predict_by_feat_single(
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                cls_logit_list=cls_logit_list,
                mlvl_priors=mlvl_priors,
                token_positive_maps=token_positive_maps,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                cls_logit_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                token_positive_maps: dict,
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = True,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            cls_logit_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            token_positive_maps (dict): Token positive map.
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

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
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        score_thr = cfg.get('score_thr', 0)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []

        for level_idx, (bbox_pred, score_factor, cls_logit, priors) in \
                enumerate(zip(bbox_pred_list,
                              score_factor_list, cls_logit_list, mlvl_priors)):
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(
                -1, self.bbox_coder.encode_size)
            score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()

            scores = convert_grounding_to_cls_scores(
                logits=cls_logit.sigmoid()[None],
                positive_maps=[token_positive_maps])[0]

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))

            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            score_factor = score_factor[keep_idxs]
            scores = torch.sqrt(scores * score_factor)

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        predictions = self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

        if len(predictions) > 0:
            # Note: GLIP adopts a very strange bbox decoder logic,
            # and if 1 is not added here, it will not align with
            # the official mAP.
            predictions.bboxes[:, 2:] = predictions.bboxes[:, 2:] + 1
        return predictions
