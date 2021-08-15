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


@HEADS.register_module()
class CenterNet2Head(BaseDenseHead, BBoxTestMixin):

    """RPN head used in `Probabilistic two-stage detection.

        <https://arxiv.org/abs/2103.07461>`_.

    Args:
        num_classes (int): No actual use in this head, but keep the
            input value for some checks of mmdet datasets and heads.
        in_channels (int): Number of channels of input features.
        feat_channels (int): Number of channels in the intermediate
            features.
        stacked_convs (int): Number of Shared convs for regression
            head and agnostic heatmap head.
        dcn_on_last_conv (bool): Whether to use deformable convolution
            layer to the last of the stacked convs.
        strides (list or tuple[int]): Strides of input features.
        regress_ranges (tuple[tuple[int, int]]): Criteria to assign
            bboxes into different feature map level.
        min_radius (int): Minimal radius for the Gaussian heatmap
            generation.
        hm_min_overlap (float): Minimal bbox overlap, together with
            min_radius to restrict Gaussian heatmap radius.
        original_dis_map (bool): Use original heatmap dist2 map.
            Apparently, the coder who is editing this is not fully
            agreed with original CenterNet2 code.
        not_norm_reg (bool): Whether to add weight to regression
            targets or not.
        loss_hm: Heatmap loss cfg.
        loss_bbox: Regression loss cfg.
        conv_cfg (dict): Covolution layer cfg.
        norm_cfg (dict): Normalization cfg.
        train_cfg (dict) Training cfg.
        init_cfg (dict or list[dict], optional): Initialization cfg.

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
                 regress_ranges=((0, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 min_radius=4,
                 hm_min_overlap=0.8,
                 original_dis_map=True,
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
                     std=0.01,
                     override=[
                         dict(
                             type='Normal',
                             name='hm_head',
                             std=0.01,
                             bias_prob=0.01),
                         dict(
                             type='Normal',
                             name='reg_head',
                             std=0.01,
                             bias=8.)])):
        super(CenterNet2Head, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.regress_ranges = regress_ranges
        self.min_radius = min_radius
        self.hm_min_overlap = hm_min_overlap
        self.original_dis_map = original_dis_map
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
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            tower.append(
                ConvModule(
                    in_channel,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.add_module('tower', nn.Sequential(*tower))
        self.hm_head = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.reg_head = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.regress_ranges])

    def simple_test_rpn(self, x, img_metas):
        """Simple forward test function."""
        hm_scores, reg_preds = self(x)
        proposal_list = self.get_bboxes(
            hm_scores, reg_preds, img_metas, cfg=self.test_cfg)
        return proposal_list

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Calculate and return losses.

        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (dict): Train / postprocessing configuration.

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
        losses = self.loss(hm_scores, reg_preds, gt_bboxes)

        if not proposal_cfg:
            proposal_cfg = self.train_cfg
        if not proposal_cfg:
            proposal_cfg = self.test_cfg
        proposal_list = self.get_bboxes(
            hm_scores, reg_preds, img_metas, cfg=proposal_cfg)
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
    def loss(self, hm_scores, reg_preds, gt_bboxes):
        """Compute losses.

        Args:
            hm_scores (list[Tensor]): Heatmap predictions for each fpn
                level, each list item has shape [B, 1, H, W].
            reg_preds (list[Tensor]): Regression predictions for each
                fpn level, each list item has shape [B, 4, H, W].
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (numberOfGtboxesInTheImage, 4).
        Returns:
            losses (dict): Losses of CenterNet2 head, including pos_loss
                neg_loss, calculated by BinaryFocalLoss and regression loss.
        """

        feature_map_sizes = [score.size()[-2:] for score in hm_scores]
        points = self.get_points(
            feature_map_sizes, reg_preds[0].dtype, reg_preds[0].device)
        hm_scores = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(-1)
             for x in hm_scores], dim=0).contiguous()
        reg_preds = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 4)
             for x in reg_preds], dim=0).contiguous()
        assert (torch.isfinite(reg_preds).all().item())
        pos_indices, reg_targets, flattened_hms = \
            self.get_targets(points, gt_bboxes)

        # Calculate regression loss
        reg_indices = (reg_targets.max(dim=1)[0] >= 0).nonzero().squeeze(1)
        reg_preds = reg_preds[reg_indices]
        reg_targets_pos = reg_targets[reg_indices]
        reg_weight_map = flattened_hms[reg_indices]
        reg_weight_map = reg_weight_map * 0 + 1 \
            if self.not_norm_reg else reg_weight_map
        reg_norm = max(reduce_mean(reg_weight_map.sum()).item(), 1)

        # Transform regression predicts into bounding boxes [x1, y1, x2, y2]
        flatten_points = torch.cat(
            [point.repeat(len(gt_bboxes), 1) for point in points])
        flatten_points = flatten_points[reg_indices]
        bbox_preds = distance2bbox(flatten_points, reg_preds)
        bbox_targets = distance2bbox(flatten_points, reg_targets_pos)
        reg_loss = self.loss_bbox(
            bbox_preds, bbox_targets, reg_weight_map, reg_norm)

        # Calculate heatmap loss
        num_pos_local = hm_scores.new_tensor(pos_indices.size(0)).float()
        num_pos_avg = max(reduce_mean(num_pos_local), 1.0)
        hm_loss = self.loss_hm(
            hm_scores, flattened_hms, pos_indices, num_pos_avg)
        losses = dict(
            pos_loss=hm_loss[0], neg_loss=hm_loss[1], loss_bbox=reg_loss)

        return losses

    def get_targets(self, points, gt_bboxes):
        """Compute predict targets for loss calculations.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (numberOfPointsInTheLevel, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (numberOfGtboxesInTheImage, 4).
        Returns:
            pos_indices (Tensor): Positive indices of flatted heatmaps in a
                batch, the shape (N) suggests the total gt_bboxes number in
                the batch.
            reg_targets (Tensor): Regression targets, the length should be
                (9*N, 4), when the neighbour 'pseudo' discrete centers are
                considered as gt_bboxes centers.
            flattened_hms (Tensor): Flattened gaussian heatmaps, the shape
                is (numberOfAllPoints * batchSize).
        """

        num_level = len(points)
        num_loc_list = [len(loc) for loc in points]
        strides = torch.cat([
            points[0].new_ones(num_loc_list[level]) * self.strides[level]
            for level in range(num_level)]).float()
        reg_size_ranges = torch.cat(
            [points[0].new_tensor(self.regress_ranges[i]).float().view(
                1, 2).expand(num_loc_list[i], 2) for i in range(num_level)])
        points = torch.cat(points, dim=0)
        m = points.shape[0]
        reg_targets = []
        flattened_hms = []
        pos_indices = []
        for i in range(len(gt_bboxes)):
            boxes = gt_bboxes[i]
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            n = boxes.shape[0]
            if n == 0:
                reg_targets.append(points.new_zeros((m, 4)) - INF)
                flattened_hms.append(points.new_zeros((m, 1)))
                continue
            left = points[:, 0].view(m, 1) - boxes[:, 0].view(1, n)
            top = points[:, 1].view(m, 1) - boxes[:, 1].view(1, n)
            right = boxes[:, 2].view(1, n) - points[:, 0].view(m, 1)
            bottom = boxes[:, 3].view(1, n) - points[:, 1].view(m, 1)
            reg_target = torch.stack([left, top, right, bottom], dim=2)

            centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2)
            centers_expanded = centers.view(1, n, 2).expand(m, n, 2)
            strides_expanded = strides.view(m, 1, 1).expand(m, n, 2)
            centers_discrete = \
                ((centers_expanded / strides_expanded).int() *
                 strides_expanded).float() + strides_expanded / 2
            is_peak = (((points.view(m, 1, 2).expand(m, n, 2) -
                         centers_discrete) ** 2).sum(dim=2) == 0)
            is_in_boxes = reg_target.min(dim=2)[0] > 0

            # Get neighbour 3x3 points
            expanded_points = points.view(m, 1, 2).expand(m, n, 2)
            dist_x = (expanded_points[:, :, 0] - centers_discrete[:, :, 0])
            dist_y = (expanded_points[:, :, 1] - centers_discrete[:, :, 1])
            is_center3x3 = (
                    (dist_x.abs() <= strides_expanded[:, :, 0])
                    & (dist_y.abs() <= strides_expanded[:, :, 1])
                    & is_in_boxes)

            # Use half diagonal length to assign boxes into different level
            # of feature maps.
            half_diagonal = ((reg_target[:, :, :2] +
                              reg_target[:, :, 2:]) ** 2).sum(dim=2) ** 0.5 / 2
            is_cared_in_the_level = (
                    (half_diagonal >= reg_size_ranges[:, [0]])
                    & (half_diagonal <= reg_size_ranges[:, [1]]))

            # Although the original author call it the 'weighted' square
            # of distance, it is more like a 'normed' one.
            dist2 = ((points.view(m, 1, 2).expand(m, n, 2) -
                      centers_expanded) ** 2).sum(dim=2)
            if self.original_dis_map:
                dist2[is_peak] = 0
            else:
                dist2[is_peak & is_cared_in_the_level] = 0
            delta = (1 - self.hm_min_overlap) / (1 + self.hm_min_overlap)
            radius2 = delta ** 2 * 2 * area
            radius2 = torch.clamp(radius2, min=self.min_radius ** 2)
            weighted_dist2 = dist2 / radius2.view(1, n).expand(m, n)

            # Get regression targets
            dist = weighted_dist2.clone()
            reg_mask = is_center3x3 & is_cared_in_the_level
            dist[reg_mask == 0] = INF * 1.0
            min_dist, min_inds = dist.min(dim=1)
            reg_targets_per_im = reg_target[range(len(reg_target)), min_inds]
            reg_targets_per_im[min_dist == INF] = - INF
            reg_targets.append(reg_targets_per_im)

            # Get heatmap targets
            pos_index = is_peak & is_cared_in_the_level
            pos_indices.append(pos_index.any(-1))
            heatmap = torch.exp(-weighted_dist2.min(dim=1)[0])
            heatmap[heatmap < 1e-4] = 0
            flattened_hms.append(heatmap)

        # Transpose image first training_targets to level first ones
        reg_targets_list = []
        flattened_hms_list = []
        pos_indices_list = []
        for im_idx in range(len(reg_targets)):
            reg_targets[im_idx] = reg_targets[im_idx].split(num_loc_list)
            flattened_hms[im_idx] = flattened_hms[im_idx].split(num_loc_list)
            pos_indices[im_idx] = pos_indices[im_idx].split(num_loc_list)
        for reg_targets_per_level in zip(*reg_targets):
            reg_targets_list.append(torch.cat(reg_targets_per_level))
        for hm_targets_per_level in zip(*flattened_hms):
            flattened_hms_list.append(torch.cat(hm_targets_per_level))
        for pos_indices_per_level in zip(*pos_indices):
            pos_indices_list.append(torch.cat(pos_indices_per_level))
        for i in range(len(reg_targets_list)):
            reg_targets_list[i] /= float(self.strides[i])
        reg_targets = torch.cat([x for x in reg_targets_list])
        flattened_hms = torch.cat([x for x in flattened_hms_list])
        pos_indices = torch.cat([x for x in pos_indices_list])
        pos_indices = torch.tensor(
            range(pos_indices.size(0))).long()[pos_indices]
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
                hm_scores[i][img_id].detach().sigmoid_()
                for i in range(num_levels)]
            bbox_pred_list = [reg_preds[i][img_id].detach()
                              for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(
                hm_score_list, bbox_pred_list, points,
                img_shape, scale_factor, cfg, rescale)
            proposal_list.append(proposals)
        return proposal_list

    def _get_bboxes_single(
            self,
            hm_scores,
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
                with shape (numOfTotalPoints, 2).
            img_shapes (list[tuple[int]]): Image shape, (height, width, C).
            scale_factors ([ndarray]): Scale factor of the image arrange as
                array[w_scale, h_scale, w_scale, h_scale].
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            proposal (Tensor): proposals for a single image, Tensor shape
                (PN, 5) where PN is the number of all proposals for a single
                image.
        """

        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        for level in range(len(hm_scores)):
            hm_score = hm_scores[level].permute(1, 2, 0).reshape(-1)
            reg_pred = reg_preds[level].permute(1, 2, 0).reshape(-1, 4)
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
                hm_score = ranked_scores[:pre_nms_top_n]
            reg_pred *= self.strides[level]
            bboxes = distance2bbox(point, reg_pred, max_shape=img_shapes)
            mlvl_scores.append(hm_score)
            mlvl_bbox_preds.append(bboxes)
            level_ids.append(
                hm_score.new_full(
                    (hm_score.size(0),), level, dtype=torch.long))
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
        scores = torch.sqrt(scores)
        if rescale:
            bboxes /= bboxes.new_tensor(scale_factors).unsqueeze(1)
        if with_nms and scores.size(0) > cfg.nms.get('max_num'):
            proposals, _ = batched_nms(bboxes, scores, nms_indices, cfg.nms)
        else:
            proposals = torch.cat([bboxes, scores[:, None]], dim=-1)
        return proposals

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
            flatten (bool): Whether to flat the points or not
        Returns:
            mlvl_points (tuple): points of each image.
        """

        mlvl_points = []
        for i in range(len(featmap_sizes)):
            h, w = featmap_sizes[i]
            stride = self.strides[i]
            x_range = torch.arange(w, device=device).to(dtype)
            y_range = torch.arange(h, device=device).to(dtype)
            y, x = torch.meshgrid(y_range, x_range)
            points = torch.stack(
                (x.reshape(-1) * stride, y.reshape(-1) * stride),
                dim=-1) + stride // 2
            mlvl_points.append(points)
        return mlvl_points

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """

        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
