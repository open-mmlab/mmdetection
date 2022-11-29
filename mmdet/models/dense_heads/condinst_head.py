# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, kaiming_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..task_modules.prior_generators import MlvlPointGenerator
from ..utils import (aligned_bilinear, filter_scores_and_topk, multi_apply,
                     relative_coordinate_maps, select_single_mlvl)
from ..utils.misc import empty_instances
from .base_mask_head import BaseMaskHead
from .fcos_head import FCOSHead

INF = 1e8


@MODELS.register_module()
class CondInstBboxHead(FCOSHead):
    """CondInst box head used in https://arxiv.org/abs/1904.02689.

    Note that CondInst Bbox Head is a extension of FCOS head.
    Two differences are described as follows:

    1. CondInst box head predicts a set of params for each instance.
    2. CondInst box head return the pos_gt_inds and pos_inds.

    Args:
        num_params (int): Number of params for instance segmentation.
    """

    def __init__(self, *args, num_params: int = 169, **kwargs) -> None:
        self.num_params = num_params
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self.controller = nn.Conv2d(
            self.feat_channels, self.num_params, 3, padding=1)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions, centerness
            predictions and param predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = \
            super(FCOSHead, self).forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        param_pred = self.controller(reg_feat)
        return cls_score, bbox_pred, centerness, param_pred

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        param_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            param_preds (List[Tensor]): param_pred for each scale level, each
                is a 4D-tensor, the channel number is num_params.
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
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # Need stride for rel coord compute
        all_level_points_strides = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device,
            with_stride=True)
        all_level_points = [i[:, :2] for i in all_level_points_strides]
        all_level_strides = [i[:, 2] for i in all_level_points_strides]
        labels, bbox_targets, pos_inds_list, pos_gt_inds_list = \
            self.get_targets(all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        self._raw_positive_infos.update(cls_scores=cls_scores)
        self._raw_positive_infos.update(centernesses=centernesses)
        self._raw_positive_infos.update(param_preds=param_preds)
        self._raw_positive_infos.update(all_level_points=all_level_points)
        self._raw_positive_infos.update(all_level_strides=all_level_strides)
        self._raw_positive_infos.update(pos_gt_inds_list=pos_gt_inds_list)
        self._raw_positive_infos.update(pos_inds_list=pos_inds_list)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    def get_targets(
        self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
            level.
            - pos_inds_list (list[Tensor]): pos_inds of each image.
            - pos_gt_inds_list (List[Tensor]): pos_gt_inds of each image.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, pos_inds_list, pos_gt_inds_list = \
            multi_apply(
                self._get_targets_single,
                batch_gt_instances,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets, pos_inds_list,
                pos_gt_inds_list)

    def _get_targets_single(
        self, gt_instances: InstanceData, points: Tensor,
        regress_ranges: Tensor, num_points_per_lvl: List[int]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.get('masks', None)

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((0,), dtype=torch.int64), \
                   gt_bboxes.new_zeros((0,), dtype=torch.int64)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            # if gt_mask not None, use gt mask's centroid to determine
            # the center region rather than gt_bbox center
            if gt_masks is None:
                center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
                center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            else:
                h, w = gt_masks.height, gt_masks.width
                masks = gt_masks.to_tensor(
                    dtype=torch.bool, device=gt_bboxes.device)
                yys = torch.arange(
                    0, h, dtype=torch.float32, device=masks.device)
                xxs = torch.arange(
                    0, w, dtype=torch.float32, device=masks.device)
                # m00/m10/m01 represent the moments of a contour
                # centroid is computed by m00/m10 and m00/m01
                m00 = masks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
                m10 = (masks * xxs).sum(dim=-1).sum(dim=-1)
                m01 = (masks * yys[:, None]).sum(dim=-1).sum(dim=-1)
                center_xs = m10 / m00
                center_ys = m01 / m00

                center_xs = center_xs[None].expand(num_points, num_gts)
                center_ys = center_ys[None].expand(num_points, num_gts)
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        # return pos_inds & pos_gt_inds
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().reshape(-1)
        pos_gt_inds = min_area_inds[labels < self.num_classes]
        return labels, bbox_targets, pos_inds, pos_gt_inds

    def get_positive_infos(self) -> InstanceList:
        """Get positive information from sampling results.

        Returns:
            list[:obj:`InstanceData`]: Positive information of each image,
            usually including positive bboxes, positive labels, positive
            priors, etc.
        """
        assert len(self._raw_positive_infos) > 0

        pos_gt_inds_list = self._raw_positive_infos['pos_gt_inds_list']
        pos_inds_list = self._raw_positive_infos['pos_inds_list']
        num_imgs = len(pos_gt_inds_list)

        cls_score_list = []
        centerness_list = []
        param_pred_list = []
        point_list = []
        stride_list = []
        for cls_score_per_lvl, centerness_per_lvl, param_pred_per_lvl,\
            point_per_lvl, stride_per_lvl in \
            zip(self._raw_positive_infos['cls_scores'],
                self._raw_positive_infos['centernesses'],
                self._raw_positive_infos['param_preds'],
                self._raw_positive_infos['all_level_points'],
                self._raw_positive_infos['all_level_strides']):
            cls_score_per_lvl = \
                cls_score_per_lvl.permute(
                    0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            centerness_per_lvl = \
                centerness_per_lvl.permute(
                    0, 2, 3, 1).reshape(num_imgs, -1, 1)
            param_pred_per_lvl = \
                param_pred_per_lvl.permute(
                    0, 2, 3, 1).reshape(num_imgs, -1, self.num_params)
            point_per_lvl = point_per_lvl.unsqueeze(0).repeat(num_imgs, 1, 1)
            stride_per_lvl = stride_per_lvl.unsqueeze(0).repeat(num_imgs, 1)

            cls_score_list.append(cls_score_per_lvl)
            centerness_list.append(centerness_per_lvl)
            param_pred_list.append(param_pred_per_lvl)
            point_list.append(point_per_lvl)
            stride_list.append(stride_per_lvl)
        cls_scores = torch.cat(cls_score_list, dim=1)
        centernesses = torch.cat(centerness_list, dim=1)
        param_preds = torch.cat(param_pred_list, dim=1)
        all_points = torch.cat(point_list, dim=1)
        all_strides = torch.cat(stride_list, dim=1)

        positive_infos = []
        for i, (pos_gt_inds,
                pos_inds) in enumerate(zip(pos_gt_inds_list, pos_inds_list)):
            pos_info = InstanceData()
            pos_info.points = all_points[i][pos_inds]
            pos_info.strides = all_strides[i][pos_inds]
            pos_info.scores = cls_scores[i][pos_inds]
            pos_info.centernesses = centernesses[i][pos_inds]
            pos_info.param_preds = param_preds[i][pos_inds]
            pos_info.pos_assigned_gt_inds = pos_gt_inds
            pos_info.pos_inds = pos_inds
            positive_infos.append(pos_info)
        return positive_infos

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        param_preds: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            param_preds (list[Tensor], optional): Params for all scale
                level, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_params, H, W)
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
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
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        all_level_points_strides = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device,
            with_stride=True)
        all_level_points = [i[:, :2] for i in all_level_points_strides]
        all_level_strides = [i[:, 2] for i in all_level_points_strides]

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]
            param_pred_list = select_single_mlvl(
                param_preds, img_id, detach=True)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                param_pred_list=param_pred_list,
                mlvl_points=all_level_points,
                mlvl_strides=all_level_strides,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                param_pred_list: List[Tensor],
                                mlvl_points: List[Tensor],
                                mlvl_strides: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
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
            param_pred_list (List[Tensor]): Param predition from all scale
                levels of a single image, each item has shape
                (num_priors * num_params, H, W).
            mlvl_points (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid.
                It has shape (num_priors, 2)
            mlvl_strides (List[Tensor]):  Each element in the list is
                the stride of a single level in feature pyramid.
                It has shape (num_priors, 1)
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
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_param_preds = []
        mlvl_valid_points = []
        mlvl_valid_strides = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor,
                        param_pred, points, strides) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, param_pred_list,
                              mlvl_points, mlvl_strides)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            param_pred = param_pred.permute(1, 2,
                                            0).reshape(-1, self.num_params)

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred,
                    param_pred=param_pred,
                    points=points,
                    strides=strides))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            param_pred = filtered_results['param_pred']
            points = filtered_results['points']
            strides = filtered_results['strides']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_param_preds.append(param_pred)
            mlvl_valid_points.append(points)
            mlvl_valid_strides.append(strides)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_points)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        results.param_preds = torch.cat(mlvl_param_preds)
        results.points = torch.cat(mlvl_valid_points)
        results.strides = torch.cat(mlvl_valid_strides)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)


class MaskFeatModule(BaseModule):
    """CondInst mask feature map branch used in \
    https://arxiv.org/abs/1904.02689.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
             map branch.
        start_level (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.
        end_level (int): The ending feature map level from rpn that
             will be used to predict the mask feature map.
        out_channels (int): Number of output channels of the mask feature
             map branch. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
        mask_stride (int): Downsample factor of the mask feature map output.
            Defaults to 4.
        num_stacked_convs (int): Number of convs in mask feature branch.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 start_level: int,
                 end_level: int,
                 out_channels: int,
                 mask_stride: int = 4,
                 num_stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = [
                     dict(type='Normal', layer='Conv2d', std=0.01)
                 ],
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        self.mask_stride = mask_stride
        self.num_stacked_convs = num_stacked_convs
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            convs_per_level.add_module(
                f'conv{i}',
                ConvModule(
                    self.in_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False,
                    bias=False))
            self.convs_all_levels.append(convs_per_level)

        conv_branch = []
        for _ in range(self.num_stacked_convs):
            conv_branch.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=False))
        self.conv_branch = nn.Sequential(*conv_branch)

        self.conv_pred = nn.Conv2d(
            self.feat_channels, self.out_channels, 1, stride=1)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()
        kaiming_init(self.convs_all_levels, a=1, distribution='uniform')
        kaiming_init(self.conv_branch, a=1, distribution='uniform')
        kaiming_init(self.conv_pred, a=1, distribution='uniform')

    def forward(self, x: Tuple[Tensor]) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            Tensor: The predicted mask feature map.
        """
        inputs = x[self.start_level:self.end_level + 1]
        assert len(inputs) == (self.end_level - self.start_level + 1)
        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        target_h, target_w = feature_add_all_level.size()[2:]
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            x_p = self.convs_all_levels[i](input_p)
            h, w = x_p.size()[2:]
            factor_h = target_h // h
            factor_w = target_w // w
            assert factor_h == factor_w
            feature_per_level = aligned_bilinear(x_p, factor_h)
            feature_add_all_level = feature_add_all_level + \
                feature_per_level

        feature_add_all_level = self.conv_branch(feature_add_all_level)
        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred


@MODELS.register_module()
class CondInstMaskHead(BaseMaskHead):
    """CondInst mask head used in https://arxiv.org/abs/1904.02689.

    This head outputs the mask for CondInst.

    Args:
        mask_feature_head (dict): Config of CondInstMaskFeatHead.
        num_layers (int): Number of dynamic conv layers.
        feat_channels (int): Number of channels in the dynamic conv.
        mask_out_stride (int): The stride of the mask feat.
        size_of_interest (int): The size of the region used in rel coord.
        max_masks_to_train (int): Maximum number of masks to train for
            each image.
        loss_segm (:obj:`ConfigDict` or dict, optional): Config of
            segmentation loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config
            of head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            head.
    """

    def __init__(self,
                 mask_feature_head: ConfigType,
                 num_layers: int = 3,
                 feat_channels: int = 8,
                 mask_out_stride: int = 4,
                 size_of_interest: int = 8,
                 max_masks_to_train: int = -1,
                 topk_masks_per_img: int = -1,
                 loss_mask: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.mask_feature_head = MaskFeatModule(**mask_feature_head)
        self.mask_feat_stride = self.mask_feature_head.mask_stride
        self.in_channels = self.mask_feature_head.out_channels
        self.num_layers = num_layers
        self.feat_channels = feat_channels
        self.size_of_interest = size_of_interest
        self.mask_out_stride = mask_out_stride
        self.max_masks_to_train = max_masks_to_train
        self.topk_masks_per_img = topk_masks_per_img
        self.prior_generator = MlvlPointGenerator([self.mask_feat_stride])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_mask = MODELS.build(loss_mask)
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        weight_nums, bias_nums = [], []
        for i in range(self.num_layers):
            if i == 0:
                weight_nums.append((self.in_channels + 2) * self.feat_channels)
                bias_nums.append(self.feat_channels)
            elif i == self.num_layers - 1:
                weight_nums.append(self.feat_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.feat_channels * self.feat_channels)
                bias_nums.append(self.feat_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_params = sum(weight_nums) + sum(bias_nums)

    def parse_dynamic_params(
            self, params: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """parse the dynamic params for dynamic conv."""
        num_insts = params.size(0)
        params_splits = list(
            torch.split_with_sizes(
                params, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:self.num_layers]
        bias_splits = params_splits[self.num_layers:]
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                weight_splits[i] = weight_splits[i].reshape(
                    num_insts * self.in_channels, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(num_insts *
                                                        self.in_channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[i] = weight_splits[i].reshape(
                    num_insts * 1, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(num_insts)

        return weight_splits, bias_splits

    def dynamic_conv_forward(self, features: Tensor, weights: List[Tensor],
                             biases: List[Tensor], num_insts: int) -> Tensor:
        """dynamic forward, each layer follow a relu."""
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x: tuple, positive_infos: InstanceList) -> tuple:
        """Forward feature from the upstream network to get prototypes and
        linearly combine the prototypes, using masks coefficients, into
        instance masks. Finally, crop the instance masks with given bboxes.

        Args:
            x (Tuple[Tensor]): Feature from the upstream network, which is
                a 4D-tensor.
            positive_infos (List[:obj:``InstanceData``]): Positive information
                that calculate from detect head.

        Returns:
            tuple: Predicted instance segmentation masks
        """
        mask_feats = self.mask_feature_head(x)
        return multi_apply(self.forward_single, mask_feats, positive_infos)

    def forward_single(self, mask_feat: Tensor,
                       positive_info: InstanceData) -> Tensor:
        """Forward features of a each image."""
        pos_param_preds = positive_info.get('param_preds')
        pos_points = positive_info.get('points')
        pos_strides = positive_info.get('strides')

        num_inst = pos_param_preds.shape[0]
        mask_feat = mask_feat[None].repeat(num_inst, 1, 1, 1)
        _, _, H, W = mask_feat.size()
        if num_inst == 0:
            return (pos_param_preds.new_zeros((0, 1, H, W)), )

        locations = self.prior_generator.single_level_grid_priors(
            mask_feat.size()[2:], 0, device=mask_feat.device)

        rel_coords = relative_coordinate_maps(locations, pos_points,
                                              pos_strides,
                                              self.size_of_interest,
                                              mask_feat.size()[2:])
        mask_head_inputs = torch.cat([rel_coords, mask_feat], dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = self.parse_dynamic_params(pos_param_preds)
        mask_preds = self.dynamic_conv_forward(mask_head_inputs, weights,
                                               biases, num_inst)
        mask_preds = mask_preds.reshape(-1, H, W)
        mask_preds = aligned_bilinear(
            mask_preds.unsqueeze(0),
            int(self.mask_feat_stride / self.mask_out_stride)).squeeze(0)

        return (mask_preds, )

    def loss_by_feat(self, mask_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict], positive_infos: InstanceList,
                     **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (list[Tensor]): List of predicted masks, each has
                shape (num_classes, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``masks``,
                and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of multiple images.
            positive_infos (List[:obj:``InstanceData``]): Information of
                positive samples of each image that are assigned in detection
                head.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert positive_infos is not None, \
            'positive_infos should not be None in `CondInstMaskHead`'
        losses = dict()

        loss_mask = 0.
        num_imgs = len(mask_preds)
        total_pos = 0

        for idx in range(num_imgs):
            (mask_pred, pos_mask_targets, num_pos) = \
                self._get_targets_single(
                mask_preds[idx], batch_gt_instances[idx],
                positive_infos[idx])
            # mask loss
            total_pos += num_pos
            if num_pos == 0 or pos_mask_targets is None:
                loss = mask_pred.new_zeros(1).mean()
            else:
                loss = self.loss_mask(
                    mask_pred, pos_mask_targets,
                    reduction_override='none').sum()
            loss_mask += loss

        if total_pos == 0:
            total_pos += 1  # avoid nan
        loss_mask = loss_mask / total_pos
        losses.update(loss_mask=loss_mask)
        return losses

    def _get_targets_single(self, mask_preds: Tensor,
                            gt_instances: InstanceData,
                            positive_info: InstanceData):
        """Compute targets for predictions of single image.

        Args:
            mask_preds (Tensor): Predicted prototypes with shape
                (num_classes, H, W).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes``, ``labels``,
                and ``masks`` attributes.
            positive_info (:obj:`InstanceData`): Information of positive
                samples that are assigned in detection head. It usually
                contains following keys.

                    - pos_assigned_gt_inds (Tensor): Assigner GT indexes of
                      positive proposals, has shape (num_pos, )
                    - pos_inds (Tensor): Positive index of image, has
                      shape (num_pos, ).
                    - param_pred (Tensor): Positive param preditions
                      with shape (num_pos, num_params).

        Returns:
            tuple: Usually returns a tuple containing learning targets.

            - mask_preds (Tensor): Positive predicted mask with shape
              (num_pos, mask_h, mask_w).
            - pos_mask_targets (Tensor): Positive mask targets with shape
              (num_pos, mask_h, mask_w).
            - num_pos (int): Positive numbers.
        """
        gt_bboxes = gt_instances.bboxes
        device = gt_bboxes.device
        gt_masks = gt_instances.masks.to_tensor(
            dtype=torch.bool, device=device).float()

        # process with mask targets
        pos_assigned_gt_inds = positive_info.get('pos_assigned_gt_inds')
        scores = positive_info.get('scores')
        centernesses = positive_info.get('centernesses')
        num_pos = pos_assigned_gt_inds.size(0)

        if gt_masks.size(0) == 0 or num_pos == 0:
            return mask_preds, None, 0
        # Since we're producing (near) full image masks,
        # it'd take too much vram to backprop on every single mask.
        # Thus we select only a subset.
        if (self.max_masks_to_train != -1) and \
           (num_pos > self.max_masks_to_train):
            perm = torch.randperm(num_pos)
            select = perm[:self.max_masks_to_train]
            mask_preds = mask_preds[select]
            pos_assigned_gt_inds = pos_assigned_gt_inds[select]
            num_pos = self.max_masks_to_train
        elif self.topk_masks_per_img != -1:
            unique_gt_inds = pos_assigned_gt_inds.unique()
            num_inst_per_gt = max(
                int(self.topk_masks_per_img / len(unique_gt_inds)), 1)

            keep_mask_preds = []
            keep_pos_assigned_gt_inds = []
            for gt_ind in unique_gt_inds:
                per_inst_pos_inds = (pos_assigned_gt_inds == gt_ind)
                mask_preds_per_inst = mask_preds[per_inst_pos_inds]
                gt_inds_per_inst = pos_assigned_gt_inds[per_inst_pos_inds]
                if sum(per_inst_pos_inds) > num_inst_per_gt:
                    per_inst_scores = scores[per_inst_pos_inds].sigmoid().max(
                        dim=1)[0]
                    per_inst_centerness = centernesses[
                        per_inst_pos_inds].sigmoid().reshape(-1, )
                    select = (per_inst_scores * per_inst_centerness).topk(
                        k=num_inst_per_gt, dim=0)[1]
                    mask_preds_per_inst = mask_preds_per_inst[select]
                    gt_inds_per_inst = gt_inds_per_inst[select]
                keep_mask_preds.append(mask_preds_per_inst)
                keep_pos_assigned_gt_inds.append(gt_inds_per_inst)
            mask_preds = torch.cat(keep_mask_preds)
            pos_assigned_gt_inds = torch.cat(keep_pos_assigned_gt_inds)
            num_pos = pos_assigned_gt_inds.size(0)

        # Follow the origin implement
        start = int(self.mask_out_stride // 2)
        gt_masks = gt_masks[:, start::self.mask_out_stride,
                            start::self.mask_out_stride]
        gt_masks = gt_masks.gt(0.5).float()
        pos_mask_targets = gt_masks[pos_assigned_gt_inds]

        return (mask_preds, pos_mask_targets, num_pos)

    def predict_by_feat(self,
                        mask_preds: List[Tensor],
                        results_list: InstanceList,
                        batch_img_metas: List[dict],
                        rescale: bool = True,
                        **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        mask results.

        Args:
            mask_preds (list[Tensor]): Predicted prototypes with shape
                (num_classes, H, W).
            results_list (List[:obj:``InstanceData``]): BBoxHead results.
            batch_img_metas (list[dict]): Meta information of all images.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Processed results of multiple
            images.Each :obj:`InstanceData` usually contains
            following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        assert len(mask_preds) == len(results_list) == len(batch_img_metas)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            bboxes = results.bboxes
            mask_pred = mask_preds[img_id]
            if bboxes.shape[0] == 0 or mask_pred.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type='mask',
                    instance_results=[results])[0]
            else:
                im_mask = self._predict_by_feat_single(
                    mask_preds=mask_pred,
                    bboxes=bboxes,
                    img_meta=img_meta,
                    rescale=rescale)
                results.masks = im_mask
        return results_list

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                img_meta: dict,
                                rescale: bool,
                                cfg: OptConfigType = None):
        """Transform a single image's features extracted from the head into
        mask results.

        Args:
            mask_preds (Tensor): Predicted prototypes, has shape [H, W, N].
            img_meta (dict): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If rescale is False, then returned masks will
                fit the scale of imgs[0].
            cfg (dict, optional): Config used in test phase.
                Defaults to None.

        Returns:
            :obj:`InstanceData`: Processed results of single image.
             it usually contains following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        cfg = self.test_cfg if cfg is None else cfg
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['img_shape'][:2]
        ori_h, ori_w = img_meta['ori_shape'][:2]

        mask_preds = mask_preds.sigmoid().unsqueeze(0)
        mask_preds = aligned_bilinear(mask_preds, self.mask_out_stride)
        mask_preds = mask_preds[:, :, :img_h, :img_w]
        if rescale:  # in-placed rescale the bboxes
            scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
                (1, 2))
            bboxes /= scale_factor

            masks = F.interpolate(
                mask_preds, (ori_h, ori_w),
                mode='bilinear',
                align_corners=False).squeeze(0) > cfg.mask_thr
        else:
            masks = mask_preds.squeeze(0) > cfg.mask_thr

        return masks
