# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptInstanceList, OptMultiConfig
from ..utils import filter_scores_and_topk, multi_apply
from .anchor_free_head import AnchorFreeHead

INF = 1e8


class FeatureAlign(BaseModule):
    """Feature Align Module.

    Feature Align Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deform conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Size of the convolution kernel.
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        deform_groups: (int): Group number of DCN in
            FeatureAdaption module.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        deform_groups: int = 4,
        init_cfg: OptMultiConfig = dict(
            type='Normal',
            layer='Conv2d',
            std=0.1,
            override=dict(type='Normal', name='conv_adaption', std=0.01))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            4, deform_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, shape: Tensor) -> Tensor:
        """Forward function of feature align module.

        Args:
            x (Tensor): Features from the upstream network.
            shape (Tensor): Exponential of bbox predictions.

        Returns:
            x (Tensor): The aligned features.
        """
        offset = self.conv_offset(shape)
        x = self.relu(self.conv_adaption(x, offset))
        return x


@MODELS.register_module()
class FoveaHead(AnchorFreeHead):
    """Detection Head of `FoveaBox: Beyond Anchor-based Object Detector.

    <https://arxiv.org/abs/1904.03797>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        base_edge_list (list[int]): List of edges.
        scale_ranges (list[tuple]): Range of scales.
        sigma (float): Super parameter of ``FoveaHead``.
        with_deform (bool):  Whether use deform conv.
        deform_groups (int): Deformable conv group size.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 base_edge_list: List[int] = (16, 32, 64, 128, 256),
                 scale_ranges: List[tuple] = ((8, 32), (16, 64), (32, 128),
                                              (64, 256), (128, 512)),
                 sigma: float = 0.4,
                 with_deform: bool = False,
                 deform_groups: int = 4,
                 init_cfg: OptMultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.with_deform = with_deform
        self.deform_groups = deform_groups
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        # box branch
        super()._init_reg_convs()
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        # cls branch
        if not self.with_deform:
            super()._init_cls_convs()
            self.conv_cls = nn.Conv2d(
                self.feat_channels, self.cls_out_channels, 3, padding=1)
        else:
            self.cls_convs = nn.ModuleList()
            self.cls_convs.append(
                ConvModule(
                    self.feat_channels, (self.feat_channels * 4),
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.cls_convs.append(
                ConvModule((self.feat_channels * 4), (self.feat_channels * 4),
                           1,
                           stride=1,
                           padding=0,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg,
                           bias=self.norm_cfg is None))
            self.feature_adaption = FeatureAlign(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                deform_groups=self.deform_groups)
            self.conv_cls = nn.Conv2d(
                int(self.feat_channels * 4),
                self.cls_out_channels,
                3,
                padding=1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: scores for each class and bbox predictions of input
            feature maps.
        """
        cls_feat = x
        reg_feat = x
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        if self.with_deform:
            cls_feat = self.feature_adaption(cls_feat, bbox_pred.exp())
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        return cls_score, bbox_pred

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
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
        assert len(cls_scores) == len(bbox_preds)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_labels, flatten_bbox_targets = self.get_targets(
            batch_gt_instances, featmap_sizes, priors)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < self.num_classes)).nonzero().view(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs)
        if num_pos > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_weights = pos_bbox_targets.new_ones(pos_bbox_targets.size())
            loss_bbox = self.loss_bbox(
                pos_bbox_preds,
                pos_bbox_targets,
                pos_weights,
                avg_factor=num_pos)
        else:
            loss_bbox = torch.tensor(
                0,
                dtype=flatten_bbox_preds.dtype,
                device=flatten_bbox_preds.device)
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(
            self, batch_gt_instances: InstanceList, featmap_sizes: List[tuple],
            priors_list: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Compute regression and classification for priors in multiple images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            featmap_sizes (list[tuple]): Size tuple of feature maps.
            priors_list (list[Tensor]): Priors list of each fpn level, each has
                shape (num_priors, 2).

        Returns:
            tuple: Targets of each level.

            - flatten_labels (list[Tensor]): Labels of each level.
            - flatten_bbox_targets (list[Tensor]): BBox targets of each
              level.
        """
        label_list, bbox_target_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            featmap_size_list=featmap_sizes,
            priors_list=priors_list)
        flatten_labels = [
            torch.cat([
                labels_level_img.flatten() for labels_level_img in labels_level
            ]) for labels_level in zip(*label_list)
        ]
        flatten_bbox_targets = [
            torch.cat([
                bbox_targets_level_img.reshape(-1, 4)
                for bbox_targets_level_img in bbox_targets_level
            ]) for bbox_targets_level in zip(*bbox_target_list)
        ]
        flatten_labels = torch.cat(flatten_labels)
        flatten_bbox_targets = torch.cat(flatten_bbox_targets)
        return flatten_labels, flatten_bbox_targets

    def _get_targets_single(self,
                            gt_instances: InstanceData,
                            featmap_size_list: List[tuple] = None,
                            priors_list: List[Tensor] = None) -> tuple:
        """Compute regression and classification targets for a single image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            featmap_size_list (list[tuple]): Size tuple of feature maps.
            priors_list (list[Tensor]): Priors of each fpn level, each has
                shape (num_priors, 2).

        Returns:
            tuple:

            - label_list (list[Tensor]): Labels of all anchors in the image.
            - box_target_list (list[Tensor]): BBox targets of all anchors in
              the image.
        """
        gt_bboxes_raw = gt_instances.bboxes
        gt_labels_raw = gt_instances.labels
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) *
                              (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        label_list = []
        bbox_target_list = []
        # for each pyramid, find the cls and box target
        for base_len, (lower_bound, upper_bound), stride, featmap_size, \
            priors in zip(self.base_edge_list, self.scale_ranges,
                          self.strides, featmap_size_list, priors_list):
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            priors = priors.view(*featmap_size, 2)
            x, y = priors[..., 0], priors[..., 1]
            labels = gt_labels_raw.new_full(featmap_size, self.num_classes)
            bbox_targets = gt_bboxes_raw.new_ones(featmap_size[0],
                                                  featmap_size[1], 4)
            # scale assignment
            hit_indices = ((gt_areas >= lower_bound) &
                           (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                label_list.append(labels)
                bbox_target_list.append(torch.log(bbox_targets))
                continue
            _, hit_index_order = torch.sort(-gt_areas[hit_indices])
            hit_indices = hit_indices[hit_index_order]
            gt_bboxes = gt_bboxes_raw[hit_indices, :] / stride
            gt_labels = gt_labels_raw[hit_indices]
            half_w = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0])
            half_h = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            # valid fovea area: left, right, top, down
            pos_left = torch.ceil(
                gt_bboxes[:, 0] + (1 - self.sigma) * half_w - 0.5).long(). \
                clamp(0, featmap_size[1] - 1)
            pos_right = torch.floor(
                gt_bboxes[:, 0] + (1 + self.sigma) * half_w - 0.5).long(). \
                clamp(0, featmap_size[1] - 1)
            pos_top = torch.ceil(
                gt_bboxes[:, 1] + (1 - self.sigma) * half_h - 0.5).long(). \
                clamp(0, featmap_size[0] - 1)
            pos_down = torch.floor(
                gt_bboxes[:, 1] + (1 + self.sigma) * half_h - 0.5).long(). \
                clamp(0, featmap_size[0] - 1)
            for px1, py1, px2, py2, label, (gt_x1, gt_y1, gt_x2, gt_y2) in \
                    zip(pos_left, pos_top, pos_right, pos_down, gt_labels,
                        gt_bboxes_raw[hit_indices, :]):
                labels[py1:py2 + 1, px1:px2 + 1] = label
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 0] = \
                    (x[py1:py2 + 1, px1:px2 + 1] - gt_x1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 1] = \
                    (y[py1:py2 + 1, px1:px2 + 1] - gt_y1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 2] = \
                    (gt_x2 - x[py1:py2 + 1, px1:px2 + 1]) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 3] = \
                    (gt_y2 - y[py1:py2 + 1, px1:px2 + 1]) / base_len
            bbox_targets = bbox_targets.clamp(min=1. / 16, max=16.)
            label_list.append(labels)
            bbox_target_list.append(torch.log(bbox_targets))
        return label_list, bbox_target_list

    # Same as base_dense_head/_predict_by_feat_single except self._bbox_decode
    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: Optional[ConfigDict] = None,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 2).
            img_meta (dict): Image meta info.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
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
        assert len(cls_score_list) == len(bbox_pred_list)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, base_len, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, self.strides,
                              self.base_edge_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = self._bbox_decode(priors, bbox_pred, base_len, img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def _bbox_decode(self, priors: Tensor, bbox_pred: Tensor, base_len: int,
                     max_shape: int) -> Tensor:
        """Function to decode bbox.

        Args:
            priors (Tensor): Center proiors of an image, has shape
                (num_instances, 2).
            bbox_preds (Tensor): Box energies / deltas for all instances,
                has shape (batch_size, num_instances, 4).
            base_len (int): The base length.
            max_shape (int): The max shape of bbox.

        Returns:
            Tensor: Decoded bboxes in (tl_x, tl_y, br_x, br_y) format. Has
            shape (batch_size, num_instances, 4).
        """
        bbox_pred = bbox_pred.exp()

        y = priors[:, 1]
        x = priors[:, 0]
        x1 = (x - base_len * bbox_pred[:, 0]). \
            clamp(min=0, max=max_shape[1] - 1)
        y1 = (y - base_len * bbox_pred[:, 1]). \
            clamp(min=0, max=max_shape[0] - 1)
        x2 = (x + base_len * bbox_pred[:, 2]). \
            clamp(min=0, max=max_shape[1] - 1)
        y2 = (y + base_len * bbox_pred[:, 3]). \
            clamp(min=0, max=max_shape[0] - 1)
        decoded_bboxes = torch.stack([x1, y1, x2, y2], -1)
        return decoded_bboxes
