# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2distance
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..utils import multi_apply
from .anchor_free_head import AnchorFreeHead

INF = 1000000000
RangeType = Sequence[Tuple[int, int]]


def _transpose(tensor_list: List[Tensor],
               num_point_list: list) -> List[Tensor]:
    """This function is used to transpose image first tensors to level first
    ones."""
    for img_idx in range(len(tensor_list)):
        tensor_list[img_idx] = torch.split(
            tensor_list[img_idx], num_point_list, dim=0)

    tensors_level_first = []
    for targets_per_level in zip(*tensor_list):
        tensors_level_first.append(torch.cat(targets_per_level, dim=0))
    return tensors_level_first


@MODELS.register_module()
class CenterNetUpdateHead(AnchorFreeHead):
    """CenterNetUpdateHead is an improved version of CenterNet in CenterNet2.
    Paper link `<https://arxiv.org/abs/2103.07461>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channel in the input feature map.
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        hm_min_radius (int): Heatmap target minimum radius of cls branch.
            Defaults to 4.
        hm_min_overlap (float): Heatmap target minimum overlap of cls branch.
            Defaults to 0.8.
        more_pos_thresh (float): The filtering threshold when the cls branch
            adds more positive samples. Defaults to 0.2.
        more_pos_topk (int): The maximum number of additional positive samples
            added to each gt. Defaults to 9.
        soft_weight_on_reg (bool): Whether to use the soft target of the
            cls branch as the soft weight of the bbox branch.
            Defaults to False.
        loss_cls (:obj:`ConfigDict` or dict): Config of cls loss. Defaults to
            dict(type='GaussianFocalLoss', loss_weight=1.0)
        loss_bbox (:obj:`ConfigDict` or dict): Config of bbox loss. Defaults to
             dict(type='GIoULoss', loss_weight=2.0).
        norm_cfg (:obj:`ConfigDict` or dict, optional): dictionary to construct
            and config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config.
            Unused in CenterNet. Reserved for compatibility with
            SingleStageDetector.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config
            of CenterNet.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((0, 80), (64, 160), (128, 320),
                                              (256, 640), (512, INF)),
                 hm_min_radius: int = 4,
                 hm_min_overlap: float = 0.8,
                 more_pos_thresh: float = 0.2,
                 more_pos_topk: int = 9,
                 soft_weight_on_reg: bool = False,
                 loss_cls: ConfigType = dict(
                     type='GaussianFocalLoss',
                     pos_weight=0.25,
                     neg_weight=0.75,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='GIoULoss', loss_weight=2.0),
                 norm_cfg: OptConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        self.soft_weight_on_reg = soft_weight_on_reg
        self.hm_min_radius = hm_min_radius
        self.more_pos_thresh = more_pos_thresh
        self.more_pos_topk = more_pos_topk
        self.delta = (1 - hm_min_overlap) / (1 + hm_min_overlap)
        self.sigmoid_clamp = 0.0001

        # GaussianFocalLoss must be sigmoid mode
        self.use_sigmoid_cls = True
        self.cls_out_channels = num_classes

        self.regress_ranges = regress_ranges
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def _init_predictor(self) -> None:
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.num_classes, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is 4.
        """
        return multi_apply(self.forward_single, x, self.scales, self.strides)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps.

        Returns:
            tuple: scores for each class, bbox predictions of
            input feature maps.
        """
        cls_score, bbox_pred, _, _ = super().forward_single(x)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        # bbox_pred needed for gradient computation has been modified
        # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
        # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
        bbox_pred = bbox_pred.clamp(min=0)
        if not self.training:
            bbox_pred *= stride
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
                each is a 4D-tensor, the channel number is num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = cls_scores[0].size(0)
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        # 1 flatten outputs
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

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        assert (torch.isfinite(flatten_bbox_preds).all().item())

        # 2 calc reg and cls branch targets
        cls_targets, bbox_targets = self.get_targets(all_level_points,
                                                     batch_gt_instances)

        # 3 add more pos index for cls branch
        featmap_sizes = flatten_points.new_tensor(featmap_sizes)
        pos_inds, cls_labels = self.add_cls_pos_inds(flatten_points,
                                                     flatten_bbox_preds,
                                                     featmap_sizes,
                                                     batch_gt_instances)

        # 4 calc cls loss
        if pos_inds is None:
            # num_gts=0
            num_pos_cls = bbox_preds[0].new_tensor(0, dtype=torch.float)
        else:
            num_pos_cls = bbox_preds[0].new_tensor(
                len(pos_inds), dtype=torch.float)
        num_pos_cls = max(reduce_mean(num_pos_cls), 1.0)
        flatten_cls_scores = flatten_cls_scores.sigmoid().clamp(
            min=self.sigmoid_clamp, max=1 - self.sigmoid_clamp)
        cls_loss = self.loss_cls(
            flatten_cls_scores,
            cls_targets,
            pos_inds=pos_inds,
            pos_labels=cls_labels,
            avg_factor=num_pos_cls)

        # 5 calc reg loss
        pos_bbox_inds = torch.nonzero(
            bbox_targets.max(dim=1)[0] >= 0).squeeze(1)
        pos_bbox_preds = flatten_bbox_preds[pos_bbox_inds]
        pos_bbox_targets = bbox_targets[pos_bbox_inds]

        bbox_weight_map = cls_targets.max(dim=1)[0]
        bbox_weight_map = bbox_weight_map[pos_bbox_inds]
        bbox_weight_map = bbox_weight_map if self.soft_weight_on_reg \
            else torch.ones_like(bbox_weight_map)
        num_pos_bbox = max(reduce_mean(bbox_weight_map.sum()), 1.0)

        if len(pos_bbox_inds) > 0:
            pos_points = flatten_points[pos_bbox_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            bbox_loss = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=bbox_weight_map,
                avg_factor=num_pos_bbox)
        else:
            bbox_loss = flatten_bbox_preds.sum() * 0

        return dict(loss_cls=cls_loss, loss_bbox=bbox_loss)

    def get_targets(
        self,
        points: List[Tensor],
        batch_gt_instances: InstanceList,
    ) -> Tuple[Tensor, Tensor]:
        """Compute classification and bbox targets for points in multiple
        images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (Tensor): Labels of all level and batch.
            - concat_lvl_bbox_targets (Tensor): BBox targets of all \
            level and batch.
        """
        assert len(points) == len(self.regress_ranges)

        num_levels = len(points)
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        concat_strides = torch.cat([
            concat_points.new_ones(num_points[i]) * self.strides[i]
            for i in range(num_levels)
        ])

        # get labels and bbox_targets of each image
        cls_targets_list, bbox_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            strides=concat_strides)

        bbox_targets_list = _transpose(bbox_targets_list, num_points)
        cls_targets_list = _transpose(cls_targets_list, num_points)
        concat_lvl_bbox_targets = torch.cat(bbox_targets_list, 0)
        concat_lvl_cls_targets = torch.cat(cls_targets_list, dim=0)
        return concat_lvl_cls_targets, concat_lvl_bbox_targets

    def _get_targets_single(self, gt_instances: InstanceData, points: Tensor,
                            regress_ranges: Tensor,
                            strides: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute classification and bbox targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:
            return gt_labels.new_full((num_points,
                                       self.num_classes),
                                      self.num_classes), \
                   gt_bboxes.new_full((num_points, 4), -1)

        # Calculate the regression tblr target corresponding to all points
        points = points[:, None].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        strides = strides[:, None, None].expand(num_points, num_gts, 2)

        bbox_target = bbox2distance(points, gt_bboxes)  # M x N x 4

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_target.min(dim=2)[0] > 0  # M x N

        # condition2: Calculate the nearest points from
        # the upper, lower, left and right ranges from
        # the center of the gt bbox
        centers = ((gt_bboxes[..., [0, 1]] + gt_bboxes[..., [2, 3]]) / 2)
        centers_discret = ((centers / strides).int() * strides).float() + \
            strides / 2

        centers_discret_dist = points - centers_discret
        dist_x = centers_discret_dist[..., 0].abs()
        dist_y = centers_discret_dist[..., 1].abs()
        inside_gt_center3x3_mask = (dist_x <= strides[..., 0]) & \
                                   (dist_y <= strides[..., 0])

        # condition3： limit the regression range for each location
        bbox_target_wh = bbox_target[..., :2] + bbox_target[..., 2:]
        crit = (bbox_target_wh**2).sum(dim=2)**0.5 / 2
        inside_fpn_level_mask = (crit >= regress_ranges[:, [0]]) & \
                                (crit <= regress_ranges[:, [1]])
        bbox_target_mask = inside_gt_bbox_mask & \
            inside_gt_center3x3_mask & \
            inside_fpn_level_mask

        # Calculate the distance weight map
        gt_center_peak_mask = ((centers_discret_dist**2).sum(dim=2) == 0)
        weighted_dist = ((points - centers)**2).sum(dim=2)  # M x N
        weighted_dist[gt_center_peak_mask] = 0

        areas = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (
            gt_bboxes[..., 3] - gt_bboxes[..., 1])
        radius = self.delta**2 * 2 * areas
        radius = torch.clamp(radius, min=self.hm_min_radius**2)
        weighted_dist = weighted_dist / radius

        # Calculate bbox_target
        bbox_weighted_dist = weighted_dist.clone()
        bbox_weighted_dist[bbox_target_mask == 0] = INF * 1.0
        min_dist, min_inds = bbox_weighted_dist.min(dim=1)
        bbox_target = bbox_target[range(len(bbox_target)),
                                  min_inds]  # M x N x 4 --> M x 4
        bbox_target[min_dist == INF] = -INF

        # Convert to feature map scale
        bbox_target /= strides[:, 0, :].repeat(1, 2)

        # Calculate cls_target
        cls_target = self._create_heatmaps_from_dist(weighted_dist, gt_labels)

        return cls_target, bbox_target

    @torch.no_grad()
    def add_cls_pos_inds(
        self, flatten_points: Tensor, flatten_bbox_preds: Tensor,
        featmap_sizes: Tensor, batch_gt_instances: InstanceList
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Provide additional adaptive positive samples to the classification
        branch.

        Args:
            flatten_points (Tensor): The point after flatten, including
                batch image and all levels. The shape is (N, 2).
            flatten_bbox_preds (Tensor): The bbox predicts after flatten,
                including batch image and all levels. The shape is (N, 4).
            featmap_sizes (Tensor): Feature map size of all layers.
                The shape is (5, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
           tuple:

           - pos_inds (Tensor): Adaptively selected positive sample index.
           - cls_labels (Tensor): Corresponding positive class label.
        """
        outputs = self._get_center3x3_region_index_targets(
            batch_gt_instances, featmap_sizes)
        cls_labels, fpn_level_masks, center3x3_inds, \
            center3x3_bbox_targets, center3x3_masks = outputs

        num_gts, total_level, K = cls_labels.shape[0], len(
            self.strides), center3x3_masks.shape[-1]

        if num_gts == 0:
            return None, None

        # The out-of-bounds index is forcibly set to 0
        # to prevent loss calculation errors
        center3x3_inds[center3x3_masks == 0] = 0
        reg_pred_center3x3 = flatten_bbox_preds[center3x3_inds]
        center3x3_points = flatten_points[center3x3_inds].view(-1, 2)

        center3x3_bbox_targets_expand = center3x3_bbox_targets.view(
            -1, 4).clamp(min=0)

        pos_decoded_bbox_preds = self.bbox_coder.decode(
            center3x3_points, reg_pred_center3x3.view(-1, 4))
        pos_decoded_target_preds = self.bbox_coder.decode(
            center3x3_points, center3x3_bbox_targets_expand)
        center3x3_bbox_loss = self.loss_bbox(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds,
            None,
            reduction_override='none').view(num_gts, total_level,
                                            K) / self.loss_bbox.loss_weight

        # Invalid index Loss set to infinity
        center3x3_bbox_loss[center3x3_masks == 0] = INF

        # 4 is the center point of the sampled 9 points, the center point
        # of gt bbox after discretization.
        # The center point of gt bbox after discretization
        # must be a positive sample, so we force its loss to be set to 0.
        center3x3_bbox_loss.view(-1, K)[fpn_level_masks.view(-1), 4] = 0
        center3x3_bbox_loss = center3x3_bbox_loss.view(num_gts, -1)

        loss_thr = torch.kthvalue(
            center3x3_bbox_loss, self.more_pos_topk, dim=1)[0]

        loss_thr[loss_thr > self.more_pos_thresh] = self.more_pos_thresh
        new_pos = center3x3_bbox_loss < loss_thr.view(num_gts, 1)
        pos_inds = center3x3_inds.view(num_gts, -1)[new_pos]
        cls_labels = cls_labels.view(num_gts,
                                     1).expand(num_gts,
                                               total_level * K)[new_pos]
        return pos_inds, cls_labels

    def _create_heatmaps_from_dist(self, weighted_dist: Tensor,
                                   cls_labels: Tensor) -> Tensor:
        """Generate heatmaps of classification branch based on weighted
        distance map."""
        heatmaps = weighted_dist.new_zeros(
            (weighted_dist.shape[0], self.num_classes))
        for c in range(self.num_classes):
            inds = (cls_labels == c)  # N
            if inds.int().sum() == 0:
                continue
            heatmaps[:, c] = torch.exp(-weighted_dist[:, inds].min(dim=1)[0])
            zeros = heatmaps[:, c] < 1e-4
            heatmaps[zeros, c] = 0
        return heatmaps

    def _get_center3x3_region_index_targets(self,
                                            bacth_gt_instances: InstanceList,
                                            shapes_per_level: Tensor) -> tuple:
        """Get the center (and the 3x3 region near center) locations and target
        of each objects."""
        cls_labels = []
        inside_fpn_level_masks = []
        center3x3_inds = []
        center3x3_masks = []
        center3x3_bbox_targets = []

        total_levels = len(self.strides)
        batch = len(bacth_gt_instances)

        shapes_per_level = shapes_per_level.long()
        area_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1])

        # Select a total of 9 positions of 3x3 in the center of the gt bbox
        # as candidate positive samples
        K = 9
        dx = shapes_per_level.new_tensor([-1, 0, 1, -1, 0, 1, -1, 0,
                                          1]).view(1, 1, K)
        dy = shapes_per_level.new_tensor([-1, -1, -1, 0, 0, 0, 1, 1,
                                          1]).view(1, 1, K)

        regress_ranges = shapes_per_level.new_tensor(self.regress_ranges).view(
            len(self.regress_ranges), 2)  # L x 2
        strides = shapes_per_level.new_tensor(self.strides)

        start_coord_pre_level = []
        _start = 0
        for level in range(total_levels):
            start_coord_pre_level.append(_start)
            _start = _start + batch * area_per_level[level]
        start_coord_pre_level = shapes_per_level.new_tensor(
            start_coord_pre_level).view(1, total_levels, 1)
        area_per_level = area_per_level.view(1, total_levels, 1)

        for im_i in range(batch):
            gt_instance = bacth_gt_instances[im_i]
            gt_bboxes = gt_instance.bboxes
            gt_labels = gt_instance.labels
            num_gts = gt_bboxes.shape[0]
            if num_gts == 0:
                continue

            cls_labels.append(gt_labels)

            gt_bboxes = gt_bboxes[:, None].expand(num_gts, total_levels, 4)
            expanded_strides = strides[None, :,
                                       None].expand(num_gts, total_levels, 2)
            expanded_regress_ranges = regress_ranges[None].expand(
                num_gts, total_levels, 2)
            expanded_shapes_per_level = shapes_per_level[None].expand(
                num_gts, total_levels, 2)

            # calc reg_target
            centers = ((gt_bboxes[..., [0, 1]] + gt_bboxes[..., [2, 3]]) / 2)
            centers_inds = (centers / expanded_strides).long()
            centers_discret = centers_inds * expanded_strides \
                + expanded_strides // 2

            bbox_target = bbox2distance(centers_discret,
                                        gt_bboxes)  # M x N x 4

            # calc inside_fpn_level_mask
            bbox_target_wh = bbox_target[..., :2] + bbox_target[..., 2:]
            crit = (bbox_target_wh**2).sum(dim=2)**0.5 / 2
            inside_fpn_level_mask = \
                (crit >= expanded_regress_ranges[..., 0]) & \
                (crit <= expanded_regress_ranges[..., 1])

            inside_gt_bbox_mask = bbox_target.min(dim=2)[0] >= 0
            inside_fpn_level_mask = inside_gt_bbox_mask & inside_fpn_level_mask
            inside_fpn_level_masks.append(inside_fpn_level_mask)

            # calc center3x3_ind and mask
            expand_ws = expanded_shapes_per_level[..., 1:2].expand(
                num_gts, total_levels, K)
            expand_hs = expanded_shapes_per_level[..., 0:1].expand(
                num_gts, total_levels, K)
            centers_inds_x = centers_inds[..., 0:1]
            centers_inds_y = centers_inds[..., 1:2]

            center3x3_idx = start_coord_pre_level + \
                im_i * area_per_level + \
                (centers_inds_y + dy) * expand_ws + \
                (centers_inds_x + dx)
            center3x3_mask = \
                ((centers_inds_y + dy) < expand_hs) & \
                ((centers_inds_y + dy) >= 0) & \
                ((centers_inds_x + dx) < expand_ws) & \
                ((centers_inds_x + dx) >= 0)

            # recalc center3x3 region reg target
            bbox_target = bbox_target / expanded_strides.repeat(1, 1, 2)
            center3x3_bbox_target = bbox_target[..., None, :].expand(
                num_gts, total_levels, K, 4).clone()
            center3x3_bbox_target[..., 0] += dx
            center3x3_bbox_target[..., 1] += dy
            center3x3_bbox_target[..., 2] -= dx
            center3x3_bbox_target[..., 3] -= dy
            # update center3x3_mask
            center3x3_mask = center3x3_mask & (
                center3x3_bbox_target.min(dim=3)[0] >= 0)  # n x L x K

            center3x3_inds.append(center3x3_idx)
            center3x3_masks.append(center3x3_mask)
            center3x3_bbox_targets.append(center3x3_bbox_target)

        if len(inside_fpn_level_masks) > 0:
            cls_labels = torch.cat(cls_labels, dim=0)
            inside_fpn_level_masks = torch.cat(inside_fpn_level_masks, dim=0)
            center3x3_inds = torch.cat(center3x3_inds, dim=0).long()
            center3x3_bbox_targets = torch.cat(center3x3_bbox_targets, dim=0)
            center3x3_masks = torch.cat(center3x3_masks, dim=0)
        else:
            cls_labels = shapes_per_level.new_zeros(0).long()
            inside_fpn_level_masks = shapes_per_level.new_zeros(
                (0, total_levels)).bool()
            center3x3_inds = shapes_per_level.new_zeros(
                (0, total_levels, K)).long()
            center3x3_bbox_targets = shapes_per_level.new_zeros(
                (0, total_levels, K, 4)).float()
            center3x3_masks = shapes_per_level.new_zeros(
                (0, total_levels, K)).bool()
        return cls_labels, inside_fpn_level_masks, center3x3_inds, \
            center3x3_bbox_targets, center3x3_masks
