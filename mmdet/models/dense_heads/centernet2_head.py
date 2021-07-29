import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32
from mmcv.ops import batched_nms

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

INF = 1e8

@HEADS.register_module()
class CenterNet2Head(BaseDenseHead, BBoxTestMixin):
    """Anchor-free head used in `CenterNet2 <>`_.

    Slightly diffrent from original head in CenterNet2,
    The differences are: 
    
    Args:
        num_classes (int): No actual use in this head, but keep the 
            input value for some checks of mmdet datasets and heads.
        in_channels (int): Number of channel in the input feature map.
        feat_channels (int): Number of channel in the intermediate feature map.
        stacked_convs (int):
        strides (list or tuple[int])
        regress_ranges (tuple[tuple[int, int]]):
        dcn_on_last_conv (bool):
        loss_cls:
        loss_bbox:
        conv_cfg (dict):
        norm_cfg (dict):
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> self = CenterNet2Head(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 dcn_on_last_conv=True,
                 strides=(8, 16, 32, 64, 128),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01))):
        super(CenterNet2Head, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.regress_ranges = regress_ranges
        self.center_sample_radius = center_sample_radius
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.norm_on_bbox = norm_on_bbox
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.dcn_on_last_conv = dcn_on_last_conv
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        tower = nn.ModuleList()
        for i in range(self.stacked_convs):
            in_channel = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            tower.append(
                ConvModule(
                    in_channel,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.add_module(f'tower', nn.Sequential(*tower))
        # agnostic_heatmap -> cls_head
        self.cls_head = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.bbox_head = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.regress_ranges])

    def simple_test_rpn(self, x, img_metas):
        """Simple forward test function."""
        cls_scores, bbox_preds = self(x)
        proposal_list = self.get_bboxes(cls_scores, bbox_preds, img_metas, cfg=self.test_cfg)
        return proposal_list

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """

        cls_scores, bbox_preds = self(x)
        losses = self.loss(cls_scores, bbox_preds, gt_bboxes,
                           img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        if self.train_cfg is None:
            proposal_cfg = self.test_cfg.rpn
        else:
            proposal_cfg = self.train_cfg
        proposal_list = self.get_bboxes(cls_scores, bbox_preds, img_metas, cfg=proposal_cfg)

        return losses, proposal_list

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
        """

        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple: scores for each class and bbox predictions \
                predictions of input feature maps.
        """

        feat = self.tower(x)
        cls_head = self.cls_head(feat)
        bbox_pred = scale(self.bbox_head(feat))
        return cls_head, F.relu(bbox_pred)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Agnostic heatmaps for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * 1.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        cls_targets_list, bbox_targets = self.get_targets(all_level_points, featmap_sizes, gt_bboxes)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, 1) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_cls_targets = torch.cat(cls_targets_list)
        flatten_bbox_targets = torch.cat(bbox_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: 0, BG cat_id: 1
        pos_inds= torch.nonzero(flatten_cls_targets != 1).reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_cls_targets, avg_factor=num_pos)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds, pos_decoded_target_preds)
        else:
            loss_bbox = pos_bbox_preds.sum()

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)


    def get_targets(self, points, feature_map_sizes, gt_bboxes_list):
        """Compute regression and classification for points in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            feature_map_sizes:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level.
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each level.
        """

        assert len(points) == len(self.regress_ranges) == len(feature_map_sizes)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get bbox_targets and cls_targets_list, of each image
        cls_targets_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        cls_targets_list = [cls_targets.split(num_points, 0) for cls_targets in cls_targets_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]

        # concat per level image
        concat_lvl_bbox_targets = []
        concat_lvl_cls_targets = []
        for i in range(num_levels):
            concat_lvl_cls_targets.append(
                torch.cat([cls_targets[i] for cls_targets in cls_targets_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_cls_targets, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, points, regress_ranges, num_points_per_lvl):
        """Compute regression for a single image."""

        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        cls_targets_list = torch.ones(num_points, dtype=torch.long, device=gt_bboxes.device)
        if num_gts == 0:
            return cls_targets_list, gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        areas = areas[None].expand(num_points, num_gts).contiguous()
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4).contiguous()
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
        center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
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

        cls_targets_list[min_area != INF] = 0  # set as FG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        return cls_targets_list, bbox_targets

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        img_shapes = [img_metas[i]['img_shape'] for i in range(cls_scores[0].shape[0])]
        scale_factors = [img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])]

        result_list = self._get_bboxes(cls_score_list, bbox_pred_list, mlvl_points,
                                       img_shapes, scale_factors, cfg, rescale, with_nms)

        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """


        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred,  points in zip(cls_scores, bbox_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                _, rank_inds = scores.sort(descending=True)
                topk_inds =rank_inds[:, :nms_pre]
                batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds).long()
                points = points[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1).unsqueeze(-1)
        padding = batch_mlvl_scores.new_zeros(batch_size, batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [tuple(mlvl_bs) for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)]
        return det_results

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """

        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i], dtype, device, flatten))

        return mlvl_points

    def _get_points_single(self, featmap_size, stride, dtype, device, flatten=False):
        """Get points of a single scale level."""

        h, w = featmap_size
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        x_range = torch.arange(w, device=device).to(dtype)
        y_range = torch.arange(h, device=device).to(dtype)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()

        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride), dim=-1) + stride // 2
        return points

