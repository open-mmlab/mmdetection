import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
import math
from mmcv.cnn import ConvModule

import warnings
from mmcv.cnn import initialize
from mmcv import ConfigDict

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from mmdet.models import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmcv.ops import batched_nms
import copy

INF = 100000000


@HEADS.register_module()
class CenterNetHeadv2(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 hotmap_min_overlap=0.8,
                 min_radius=4,
                 norm_on_bbox=True,
                 prior_prob=0.01,
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                 loss_agn_hm=dict(
                     type='HeatmapBinaryFocalLoss',
                     use_sigmoid=True,
                     alpha=2.0,
                     gamma=4.0,
                     beta=0.25,
                     sigmoid_clamp=1e-4,
                     ignore_high_fp=0.85,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 conv_bias=True,
                 init_cfg=None,
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.delta = (1 - hotmap_min_overlap) / (1 + hotmap_min_overlap)
        self.min_radius = min_radius
        self.norm_on_bbox = norm_on_bbox
        self.prior_prob = prior_prob

        super().__init__(
            num_classes,
            in_channels,
            conv_bias=conv_bias,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_agn_hm = build_loss(loss_agn_hm)

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_reg_convs()
        self._init_predictor()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.agn_hm = nn.Conv2d(self.in_channels, 1, 3, stride=1, padding=1)


    def init_weights(self):
        """Initialize the weights."""

        for layer in self.reg_convs.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        isinstance(self.conv_reg, nn.Conv2d)
        torch.nn.init.normal_(self.conv_reg.weight, std=0.01)
        torch.nn.init.constant_(self.conv_reg.bias, 8.)

        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.normal_(self.agn_hm.weight, std=0.01)
        torch.nn.init.constant_(self.agn_hm.bias, bias_value)


    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        reg_feat = x
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        agn_hms = self.agn_hm(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        bbox_pred = scale(bbox_pred).float()
        bbox_pred = F.relu(bbox_pred)

        return bbox_pred, agn_hms


    @force_fp32(apply_to=('bbox_preds', 'agn_hm_preds'))
    def loss(self,
             bbox_preds,
             agn_hm_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        shapes_per_level = all_level_points[0].new_tensor([(x.shape[2], x.shape[3]) for x in bbox_preds])

        bbox_targets,  hms_targets = self.get_targets(all_level_points, gt_bboxes.copy(), shapes_per_level)

        pos_inds = self.get_pos_inds(gt_bboxes.copy(), shapes_per_level)

        num_imgs = bbox_preds[0].size(0)
        # flatten bbox_preds and agn_hm_pred
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_agn_hm_preds = [
            agn_hm_pred.permute(0, 2, 3, 1).reshape(-1)
            for agn_hm_pred in agn_hm_preds
        ]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_agn_hm_preds = torch.cat(flatten_agn_hm_preds)

        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_hms_targets = torch.cat(hms_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        reg_inds = torch.nonzero(flatten_bbox_targets.max(dim=1)[0] >= 0).squeeze(1)
        pos_bbox_preds = flatten_bbox_preds[reg_inds]
        pos_bbox_targets = flatten_bbox_targets[reg_inds]

        reg_weight_map = flatten_hms_targets.max(dim=1)[0]
        reg_weight_map = reg_weight_map[reg_inds]
        reg_weight_map = reg_weight_map * 0 + 1

        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(reg_weight_map.sum().detach()), 1)

        pos_points = flatten_points[reg_inds]
        if len(pos_points) > 0:
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)

            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=reg_weight_map,
                avg_factor=centerness_denorm)
        else:
            loss_bbox = pos_bbox_preds.sum()

        cat_agn_heatmap = flatten_hms_targets.max(dim=1)[0]
        num_pos_local = pos_inds.numel()
        num_pos_avg = max(reduce_mean(num_pos_local), 1.0)
        loss_agn_hm = self.loss_agn_hm(
            flatten_agn_hm_preds, cat_agn_heatmap, pos_inds, avg_factor=num_pos_avg)

        return dict(
            loss_bbox=loss_bbox,
            loss_agn_hm=loss_agn_hm)

    @force_fp32(apply_to=('bbox_preds', 'agn_hm_preds'))
    def get_bboxes(self,
                   bbox_preds,
                   agn_hm_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.
        """

        assert with_nms, '``with_nms`` in RPNHead should always True'
        assert len(agn_hm_preds) == len(bbox_preds)
        num_levels = len(agn_hm_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                agn_hm_preds[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                all_level_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_points = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)

            rpn_cls_score = rpn_cls_score.reshape(-1)
            scores = rpn_cls_score.sigmoid()

            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            rpn_bbox_pred = rpn_bbox_pred*self.strides[idx]
            points = mlvl_points[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                points = points[topk_inds, :]
            mlvl_scores.append(torch.sqrt(scores))
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_points.append(points)
            level_ids.append(
                scores.new_full((scores.size(0),), idx, dtype=torch.long))
        scores = torch.cat(mlvl_scores)
        points = torch.cat(mlvl_valid_points)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = distance2bbox(points, rpn_bbox_pred, max_shape=img_shape)

        # avoid invalid boxes in RoI heads
        proposals[:, 2] = torch.max(proposals[:, 2], proposals[:, 0] + 0.01)
        proposals[:, 3] = torch.max(proposals[:, 3], proposals[:, 1] + 0.01)

        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
        if proposals.numel() > 0:
            dets, keep = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]


    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, shapes_per_level):
        """Compute regression, classification and centerness targets for points
        in multiple images.
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
        bbox_targets_list,  hms_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
            shapes_per_level=shapes_per_level)

        # split to per img, per level
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        hms_targets_list = [
            hms_targets.split(num_points, 0)
            for hms_targets in hms_targets_list
        ]

        # concat per level image
        concat_lvl_bbox_targets = []
        concat_lvl_hms_targets = []
        for i in range(num_levels):
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            hms_targets = torch.cat(
                [hms_targets[i] for hms_targets in hms_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_hms_targets.append(hms_targets)
        return concat_lvl_bbox_targets,  concat_lvl_hms_targets

    def _get_target_single(self, gt_bboxes, points, regress_ranges, num_points_per_lvl, shapes_per_level):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        if num_gts == 0:
            return torch.tensor([0]).new_full((num_points,4), - INF), \
                   gt_bboxes.new_zeros((num_points, 1))

        #areas = areas[None].repeat(num_points, 1)
        #regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes_expand = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes_expand[..., 0]
        right = gt_bboxes_expand[..., 2] - xs
        top = ys - gt_bboxes_expand[..., 1]
        bottom = gt_bboxes_expand[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(2)[0] > 0

        # condition2: filter the feature points generated by condition1
        centers = ((gt_bboxes[:, [0, 1]] + gt_bboxes[:, [2, 3]]) / 2)
        centers_expanded = centers.view(1, num_gts, 2).expand(num_points, num_gts, 2)
        strides = torch.cat([
            shapes_per_level.new_ones(num_points_per_lvl[l]) * self.strides[l] \
            for l in range(len(num_points_per_lvl))]).float()  # M
        strides_expanded = strides.view(num_points, 1, 1).expand(num_points, num_gts, 2)
        centers_discret = ((centers_expanded / strides_expanded).int() * strides_expanded).float() + strides_expanded / 2

        is_peak = (((points.view(num_points, 1, 2).expand(num_points, num_gts, 2) - centers_discret) ** 2).sum(dim=2) == 0)
        locations_expanded = points.view(num_points, 1, 2).expand(num_points, num_gts, 2)
        dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
        dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
        is_center3x3 = (dist_x <= strides_expanded[:, :, 0]) & (dist_y <= strides_expanded[:, :, 0]) & inside_gt_bbox_mask

        # condition3: assign_reg_fpn
        crit = ((bbox_targets[:, :, :2] + bbox_targets[:, :, 2:]) ** 2).sum(dim=2) ** 0.5 / 2
        is_cared_in_the_level = (crit >= regress_ranges[:, [0]]) & (crit <= regress_ranges[:, [1]])
        reg_mask = is_center3x3 & is_cared_in_the_level

        dist2 = ((points.view(num_points, 1, 2).expand(num_points, num_gts, 2) - centers_expanded) ** 2).sum(dim=2)
        dist2[is_peak] = 0
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        radius2 = self.delta ** 2 * 2 * areas
        radius2 = torch.clamp(radius2, min=self.min_radius ** 2)
        weighted_dist2 = dist2 / radius2.view(1, num_gts).expand(num_points, num_gts)

        # result1: reg_target
        weight_distance_2 = weighted_dist2.clone()
        weight_distance_2[reg_mask == 0] = INF * 1.0
        min_dist, min_inds = weight_distance_2.min(dim=1)
        reg_targets_per_im = bbox_targets[range(len(bbox_targets)), min_inds]
        reg_targets_per_im[min_dist == INF] = - INF
        reg_target = reg_targets_per_im

        # result2: agn_heatmaps
        weight_distance_2 = weighted_dist2.clone()
        flattened_hms = weight_distance_2.new_zeros((weight_distance_2.shape[0], 1))
        flattened_hms[:, 0] = torch.exp(-weight_distance_2.min(dim=1)[0])
        zeros = flattened_hms < 1e-4
        flattened_hms[zeros] = 0

        return reg_target, flattened_hms


    def get_pos_inds(self, gt_bboxes, shapes_per_level):

        pos_inds = []

        L = len(self.strides)
        B = len(gt_bboxes)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long()
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long()
        strides_default = shapes_per_level.new_tensor(self.strides).float()
        for im_i in range(B):
            bboxes = gt_bboxes[im_i]
            n = bboxes.shape[0]
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2)
            centers = centers.view(n, 1, 2).expand(n, L, 2)
            strides = strides_default.view(1, L, 1).expand(n, L, 2)
            centers_inds = (centers / strides).long()
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            pos_ind = level_bases.view(1, L).expand(n, L) + \
                       im_i * loc_per_level.view(1, L).expand(n, L) + \
                       centers_inds[:, :, 1] * Ws + \
                       centers_inds[:, :, 0]

            # condition3: assign_fpn_level
            size_ranges = bboxes.new_tensor(self.regress_ranges).view(len(self.regress_ranges), 2)
            crit = ((bboxes[:, 2:] - bboxes[:, :2]) ** 2).sum(dim=1) ** 0.5 / 2
            n, L = crit.shape[0], size_ranges.shape[0]
            crit = crit.view(n, 1).expand(n, L)
            size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
            is_cared_in_the_level =(crit >= size_ranges_expand[:, :, 0]) & (crit <= size_ranges_expand[:, :, 1])

            pos_ind = pos_ind[is_cared_in_the_level].view(-1)
            pos_inds.append(pos_ind)

        pos_inds = torch.cat(pos_inds, dim=0).long()
        return pos_inds
