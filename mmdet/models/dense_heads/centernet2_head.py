import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32
from mmcv.ops import batched_nms

from mmdet.core import distance2bbox, multi_apply,  reduce_mean
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

INF = 1e8


def _transpose(training_targets, num_loc_list):
    """
    This function is used to transpose image first training targets to
        level first ones
    :return: level first training targets
    """
    for im_i in range(len(training_targets)):
        training_targets[im_i] = torch.split(
            training_targets[im_i], num_loc_list, dim=0)

    targets_level_first = []
    for targets_per_level in zip(*training_targets):
        targets_level_first.append(
            torch.cat(targets_per_level, dim=0))
    return targets_level_first


def _create_agn_heatmaps_from_dist(dist):
    """
    dist: M x N
    return:
      heatmaps: M x 1
    """
    heatmaps = dist.new_zeros((dist.shape[0], 1))
    heatmaps[:, 0] = torch.exp(-dist.min(dim=1)[0])
    zeros = heatmaps < 1e-4
    heatmaps[zeros] = 0
    return heatmaps


def _get_reg_targets(reg_targets, dist, mask, area):
    """
      reg_targets (M x N x 4): long tensor
      dist (M x N)
      is_*: M x N
    """
    dist[mask == 0] = INF * 1.0
    min_dist, min_inds = dist.min(dim=1)  # M
    reg_targets_per_im = reg_targets[
        range(len(reg_targets)), min_inds]  # M x N x 4 --> M x 4
    reg_targets_per_im[min_dist == INF] = - INF
    return reg_targets_per_im


def assign_reg_fpn(reg_targets_per_im, size_ranges):
    """
    Inputs:
        reg_targets_per_im: M x N x 4
        size_ranges: M x 2
    """
    crit = ((reg_targets_per_im[:, :, :2] + \
             reg_targets_per_im[:, :, 2:]) ** 2).sum(dim=2) ** 0.5 / 2  # M x N
    is_cared_in_the_level = (crit >= size_ranges[:, [0]]) & \
                            (crit <= size_ranges[:, [1]])
    return is_cared_in_the_level


def get_center3x3(locations, centers, strides):
    """
    Inputs:
        locations: M x 2
        centers: N x 2
        strides: M
    """
    M, N = locations.shape[0], centers.shape[0]
    locations_expanded = locations.view(M, 1, 2).expand(M, N, 2)  # M x N x 2
    centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # M x N x 2
    strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)  # M x N
    centers_discret = ((centers_expanded / strides_expanded).int() * \
                       strides_expanded).float() + strides_expanded / 2  # M x N x 2
    dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
    dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
    return (dist_x <= strides_expanded[:, :, 0]) & \
           (dist_y <= strides_expanded[:, :, 0])


def _get_points_single(featmap_size, stride, dtype, device, flatten=False):
    """Get points of a single scale level."""
    h, w = featmap_size
    x_range = torch.arange(w, device=device).to(dtype)
    y_range = torch.arange(h, device=device).to(dtype)
    y, x = torch.meshgrid(y_range, x_range)
    if flatten:
        y = y.flatten()
        x = x.flatten()

    points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride), dim=-1) + stride // 2
    return points


def _get_bboxes_single(hm_scores,
                       reg_preds,
                       mlvl_points,
                       img_shapes,
                       scale_factors,
                       cfg,
                       rescale=False,
                       with_nms=True):
    """Transform outputs for a single image into bbox predictions.

    Args:
        hm_scores (list[Tensor]): heatmap scores of all levels for a
            single image. Tensor shape [1, H, W]
        reg_preds (list[Tensor]): regression predictions of all levels
         for a single image. Tensor shape [4, H, W]
        mlvl_points (list[Tensor]): Box reference for a single scale level
            with shape (num_total_points, 2).
        img_shapes (list[tuple[int]]): Shape of the input image,
            list[(height, width, 3)].
        scale_factors [ndarray]: Scale factor of the image arrange as
            array[w_scale, h_scale, w_scale, h_scale].
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.
            Default: False.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.

    Returns:
        proposal (Tensor): proposals for a single image,
            Tensor shape (PN, 5) where PN is the number of all proposals
            for a single image.
    """

    level_ids = []
    mlvl_scores = []
    mlvl_bbox_preds = []
    for level in range(len(hm_scores)):
        hm_score = hm_scores[level].reshape(-1)
        reg_pred = reg_preds[level].reshape(-1, 4)
        point = mlvl_points[level]

        if cfg.get('score_thr'):
            valid_idx = hm_score > cfg.get('score_thr')
            valid_idx = valid_idx.nonzero(as_tuple=True)
            hm_score = hm_score[valid_idx]
            reg_pred = reg_pred[valid_idx]
            point = point[valid_idx]

        pre_nms_top_n = hm_score.shape[0]
        assert cfg.get('nms_pre', -1) > 0, 'Must specify nms_pre'
        pre_nms_top_n = min(pre_nms_top_n, cfg.get('nms_pre'))
        if hm_score.shape[0] >= pre_nms_top_n:
            ranked_scores, rank_inds = hm_score.sort(descending=True)
            reg_pred = reg_pred[rank_inds][:pre_nms_top_n]
            point = point[rank_inds][:pre_nms_top_n, :]
            hm_score = hm_score[:pre_nms_top_n]

        bboxes = distance2bbox(point, reg_pred, max_shape=img_shapes)
        mlvl_scores.append(hm_score)
        mlvl_bbox_preds.append(bboxes)
        level_ids.append(
            hm_score.new_full((hm_score.size(0),), level, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        bboxes = torch.cat(mlvl_bbox_preds)
        nms_indices = torch.cat(level_ids)
        min_bbox_size = cfg.get('min_bbox_size', -1)
        if min_bbox_size >= 0:
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            valid_mask = (w > min_bbox_size) & (h > min_bbox_size)
            if not valid_mask.all():
                scores = scores[valid_mask]
                bboxes = bboxes[valid_mask]
                nms_indices = nms_indices[valid_mask]

    if rescale:
        bboxes /= bboxes.new_tensor(scale_factors).unsqueeze(1)

    if with_nms and scores.size(0) > cfg.nms.get('max_num'):
        proposals, _ = batched_nms(bboxes, scores, nms_indices, cfg.nms)
    else:
        proposals = torch.cat([bboxes, scores[:, None]], dim=-1)

    return proposals


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
        loss_hm:
        loss_bbox:
        conv_cfg (dict):
        norm_cfg (dict):
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> self = CenterNet2Head(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> hm_score, reg_pred = self.forward(feats)
        >>> assert len(hm_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=288,
                 stacked_convs=4,
                 dcn_on_last_conv=True,
                 strides=(8, 16, 32, 64, 128),
                 regress_ranges=((0, 80), (64, 160), (128, 320), (256, 640),
                                 (512, INF)),
                 min_radius=4,
                 hm_min_overlap=0.8,
                 not_norm_reg=False,
                 loss_hm=dict(
                     type='BinaryFocalLoss',
                     alpha=0.25,
                     beta=4,
                     gamma=2,
                     pos_weight=0.5,
                     neg_weight=0.5,
                     sigmoid_clamp=1e-4,
                     ignore_high_fp=0.85),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01)):
        super(CenterNet2Head, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.regress_ranges = regress_ranges
        self.min_radius = min_radius
        self.hm_min_overlap = hm_min_overlap
        self.not_norm_reg = not_norm_reg
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.dcn_on_last_conv = dcn_on_last_conv
        self.loss_hm = build_loss(loss_hm)
        self.loss_bbox = build_loss(loss_bbox)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        tower = nn.ModuleList()
        for i in range(self.stacked_convs):
            in_channel = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                tower.append(
                    ConvModule(
                        in_channel,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='DCNv2'),
                        norm_cfg=self.norm_cfg))
            else:
                tower.append(
                    ConvModule(
                        in_channel,
                        self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))

        self.add_module(f'tower', nn.Sequential(*tower))
        # agnostic_heatmap -> hm_head
        self.hm_head = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.reg_head = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.regress_ranges])

    def simple_test_rpn(self, x, img_metas):
        """Simple forward test function."""
        hm_scores, reg_preds = self(x)
        proposal_list = self.get_bboxes(hm_scores, reg_preds, img_metas, cfg=self.test_cfg)
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
            losses: (dict[str, Tensor]): A dictionary of loss components.
            proposal_list: list[tuple[Tensor, Tensor]]: Each item in
                result_list is 2-tuple. The first item is an (n, 5) tensor,
                where 5 represent (tl_x, tl_y, br_x, br_y, score) and the
                score between 0 and 1. The shape of the second tensor in the
                tuple is (n,), and each element represents the class label
                of the corresponding box.
        """

        hm_scores, reg_preds = self(x)
        losses = self.loss(hm_scores, reg_preds, gt_bboxes, img_metas)
        if self.train_cfg is None:
            proposal_cfg = self.test_cfg.rpn
        else:
            proposal_cfg = self.train_cfg
        proposal_list = self.get_bboxes(hm_scores, reg_preds, img_metas, cfg=proposal_cfg)

        return losses, proposal_list

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                hm_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                reg_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
        """

        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the regression prediction.

        Returns:
            tuple: scores for each class and regression predictions \
                predictions of input feature maps.
        """

        feat = self.tower(x)
        hm_head = self.hm_head(feat)
        reg_pred = scale(self.reg_head(feat))

        return hm_head, F.relu(reg_pred)

    @force_fp32(apply_to=('hm_scores', 'reg_preds'))
    def loss(self, hm_scores, reg_preds, gt_bboxes, img_metas):

        """
        Inputs:
            hm_scores
            reg_preds
            gt_bboxes
            img_metas
        """

        feature_map_sizes = [feature_map.size()[-2:] for feature_map in hm_scores]
        points = self.get_points(feature_map_sizes, reg_preds[0].dtype, reg_preds[0].device)
        hm_scores = torch.cat([x.permute(0, 2, 3, 1).reshape(-1)
                               for x in hm_scores], dim=0)
        reg_preds = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 4)
                               for x in reg_preds], dim=0)
        assert (torch.isfinite(reg_preds).all().item())
        flatten_points = torch.cat([point.repeat(len(img_metas), 1) for point in points])

        pos_indices, reg_targets, flattened_hms = \
            self.get_targets(points, feature_map_sizes, gt_bboxes)

        num_pos_local = pos_indices.numel()
        num_pos_avg = max(reduce_mean(num_pos_local), 1.0)
        reg_inds = torch.nonzero(reg_targets.max(dim=1)[0] >= 0).squeeze(1)
        reg_preds = reg_preds[reg_inds]
        flatten_points = flatten_points[reg_inds]

        reg_targets_pos = reg_targets[reg_inds]
        reg_weight_map = flattened_hms.max(dim=1)[0]
        reg_weight_map = reg_weight_map[reg_inds]
        reg_weight_map = reg_weight_map * 0 + 1 \
            if self.not_norm_reg else reg_weight_map
        reg_norm = max(reduce_mean(reg_weight_map.sum()).item(), 1)

        bbox_preds = distance2bbox(flatten_points, reg_preds)
        bbox_targets = distance2bbox(flatten_points, reg_targets_pos)
        reg_loss = self.loss_bbox(bbox_preds, bbox_targets, reg_weight_map, reg_norm)

        cat_agn_heatmap = flattened_hms.max(dim=1)[0]  # M
        pos_loss, neg_loss = self.loss_hm(hm_scores, cat_agn_heatmap, pos_indices, num_pos_avg)
        print(hm_scores.shape, cat_agn_heatmap. shape, pos_indices.shape, bbox_targets.size(0))

        return dict(pos_loss=pos_loss, neg_loss=neg_loss, loss_bbox=reg_loss)

    def get_targets(self, points, feature_map_sizes, gt_bboxes_list):
        """Compute regression and heatmap for points in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            feature_map_sizes (list[torch.Size[2]]):
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
        Returns:
            tuple:

        """

        # Get positive pixel index
        pos_indices = self._get_label_indices(gt_bboxes_list, feature_map_sizes)
        L = len(points)
        num_loc_list = [len(loc) for loc in points]
        strides = torch.cat([
            points[0].new_ones(num_loc_list[l]) * self.strides[l] for l in range(L)]).float()  # M
        reg_size_ranges = torch.cat([points[0].new_tensor(self.regress_ranges[l]).float().view(
            1, 2).expand(num_loc_list[l], 2) for l in range(L)])
        points = torch.cat(points, dim=0)  # M x 2
        M = points.shape[0]
        reg_targets = []
        flattened_hms = []
        for i in range(len(gt_bboxes_list)):
            boxes = gt_bboxes_list[i]
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            N = boxes.shape[0]
            if N == 0:
                reg_targets.append(points.new_zeros((M, 4)) - INF)
                flattened_hms.append(points.new_zeros((M, 1)))
                continue
            l = points[:, 0].view(M, 1) - boxes[:, 0].view(1, N)  # M x N
            t = points[:, 1].view(M, 1) - boxes[:, 1].view(1, N)  # M x N
            r = boxes[:, 2].view(1, N) - points[:, 0].view(M, 1)  # M x N
            b = boxes[:, 3].view(1, N) - points[:, 1].view(M, 1)  # M x N
            reg_target = torch.stack([l, t, r, b], dim=2)  # M x N x 4

            centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2)  # N x 2
            centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # M x N x 2
            strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)
            centers_discret = ((centers_expanded / strides_expanded).int() *
                               strides_expanded).float() + strides_expanded / 2  # M x N x 2

            is_peak = (((points.view(M, 1, 2).expand(M, N, 2) -
                         centers_discret) ** 2).sum(dim=2) == 0)  # M x N
            is_in_boxes = reg_target.min(dim=2)[0] > 0  # M x N
            is_center3x3 = get_center3x3(points, centers, strides) & is_in_boxes  # M x N
            is_cared_in_the_level = assign_reg_fpn(reg_target, reg_size_ranges)
            reg_mask = is_center3x3 & is_cared_in_the_level

            dist2 = ((points.view(M, 1, 2).expand(M, N, 2) -
                      centers_expanded) ** 2).sum(dim=2)  # M x N
            dist2[is_peak] = 0
            delta = (1 - self.hm_min_overlap) / (1 + self.hm_min_overlap)
            radius2 = delta ** 2 * 2 * area  # N
            radius2 = torch.clamp(radius2, min=self.min_radius ** 2)
            weighted_dist2 = dist2 / radius2.view(1, N).expand(M, N)

            reg_target = _get_reg_targets(reg_target, weighted_dist2.clone(), reg_mask, area)
            flattened_hm = _create_agn_heatmaps_from_dist(weighted_dist2.clone())

            reg_targets.append(reg_target)
            flattened_hms.append(flattened_hm)

        # transpose im first training_targets to level first ones
        reg_targets = _transpose(reg_targets, num_loc_list)
        flattened_hms = _transpose(flattened_hms, num_loc_list)

        for i in range(len(reg_targets)):
            reg_targets[i] = reg_targets[i] / float(self.strides[i])
        reg_targets = torch.cat([x for x in reg_targets], dim=0)  # MB x 4
        flattened_hms = torch.cat([x for x in flattened_hms], dim=0)  # MB x C

        return pos_indices, reg_targets, flattened_hms

    @force_fp32(apply_to=('hm_scores', 'reg_preds'))
    def get_bboxes(self,
                   hm_scores,
                   reg_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            hm_scores (list[Tensor]): Box scores for each scale level
                with shape (N, 1, H, W).
            reg_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            proposal_list (list[Tensor]): proposals for all images,
                Tensor shape (PN, 5) where PN is the number of all proposals
                for a single image.
        """

        assert with_nms, '``with_nms`` in RPNHead should always True'
        assert len(hm_scores) == len(reg_preds)
        device = reg_preds[0].device
        num_levels = len(hm_scores)
        feature_map_sizes = [hm_score.size()[-2:] for hm_score in hm_scores]
        points = self.get_points(feature_map_sizes, reg_preds[0].dtype, device)

        proposal_list = []
        for img_id in range(len(img_metas)):
            hm_score_list = [
                hm_scores[i][img_id].detach().sigmoid() for i in range(num_levels)
            ]
            bbox_pred_list = [
                reg_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = _get_bboxes_single(hm_score_list, bbox_pred_list,
                                           points, img_shape,
                                           scale_factor, cfg, rescale)
            proposal_list.append(proposals)
        return proposal_list

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
                _get_points_single(featmap_sizes[i], self.strides[i], dtype, device, flatten))
        return mlvl_points

    def _get_label_indices(self, gt_bboxes_list, feature_map_sizes):
        pos_indices = []
        L = len(self.strides)
        B = len(gt_bboxes_list)
        shapes_per_level = gt_bboxes_list[0].new_tensor(feature_map_sizes).long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long()  # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases)  # L
        strides_default = level_bases.new_tensor(self.strides).float()
        for im_i in range(B):
            bboxes = gt_bboxes_list[im_i]
            n = bboxes.shape[0]
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2)
            centers = centers.view(n, 1, 2).expand(n, L, 2)
            strides = strides_default.view(1, L, 1).expand(n, L, 2)
            centers_inds = (centers / strides).long()
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            pos_ind = level_bases.view(1, L).expand(n, L) + \
                      im_i * loc_per_level.view(1, L).expand(n, L) + \
                      centers_inds[:, :, 1] * Ws + \
                      centers_inds[:, :, 0]  # n x L
            is_cared_in_the_level = self._assign_fpn_level(bboxes)
            pos_ind = pos_ind[is_cared_in_the_level].view(-1)
            pos_indices.append(pos_ind)
        pos_indices = torch.cat(pos_indices, dim=0).long()
        return pos_indices

    def _assign_fpn_level(self, boxes):

        """
        Inputs:
            boxes: n x 4
            size_ranges: L x 2
        Return:
            is_cared_in_the_level: n x L
        """

        size_ranges = boxes.new_tensor(
            self.regress_ranges).view(len(self.regress_ranges), 2)
        crit = ((boxes[:, 2:] - boxes[:, :2]) ** 2).sum(dim=1) ** 0.5 / 2  # n
        n, L = crit.shape[0], size_ranges.shape[0]
        crit = crit.view(n, 1).expand(n, L)
        size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
        is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
                                (crit <= size_ranges_expand[:, :, 1])
        return is_cared_in_the_level
